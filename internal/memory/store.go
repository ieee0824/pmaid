package memory

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"

	memai "github.com/ieee0824/memAI-go"
	_ "modernc.org/sqlite"
)

type SQLiteStore struct {
	db *sql.DB
}

func NewSQLiteStore(dbPath string) (*SQLiteStore, error) {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}

	db.SetMaxOpenConns(1)

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS memories (
			id TEXT PRIMARY KEY,
			content TEXT NOT NULL,
			embedding TEXT,
			thread_key TEXT DEFAULT '',
			event_date TEXT DEFAULT '',
			boost REAL DEFAULT 0,
			emotional_intensity REAL DEFAULT 0
		)
	`)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("create memories table: %w", err)
	}

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS token_usage (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TEXT NOT NULL DEFAULT (datetime('now')),
			version TEXT NOT NULL DEFAULT '',
			model TEXT NOT NULL DEFAULT '',
			turn INTEGER NOT NULL DEFAULT 0,
			prompt_tokens INTEGER NOT NULL DEFAULT 0,
			completion_tokens INTEGER NOT NULL DEFAULT 0,
			total_tokens INTEGER NOT NULL DEFAULT 0
		)
	`)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("create token_usage table: %w", err)
	}

	return &SQLiteStore{db: db}, nil
}

func (s *SQLiteStore) GetMemories(ctx context.Context) ([]memai.Memory[string], error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, content, embedding, thread_key, event_date, boost, emotional_intensity
		FROM memories
	`)
	if err != nil {
		return nil, fmt.Errorf("query memories: %w", err)
	}
	defer rows.Close()

	var memories []memai.Memory[string]
	for rows.Next() {
		var m memai.Memory[string]
		var embJSON string
		err := rows.Scan(&m.ID, &m.Content, &embJSON, &m.ThreadKey, &m.EventDate, &m.Boost, &m.EmotionalIntensity)
		if err != nil {
			return nil, fmt.Errorf("scan memory: %w", err)
		}
		if embJSON != "" {
			if err := json.Unmarshal([]byte(embJSON), &m.Embedding); err != nil {
				return nil, fmt.Errorf("unmarshal embedding: %w", err)
			}
		}
		memories = append(memories, m)
	}
	return memories, rows.Err()
}

func (s *SQLiteStore) SaveMemory(ctx context.Context, m *memai.Memory[string]) error {
	embJSON, err := json.Marshal(m.Embedding)
	if err != nil {
		return fmt.Errorf("marshal embedding: %w", err)
	}

	_, err = s.db.ExecContext(ctx, `
		INSERT OR REPLACE INTO memories (id, content, embedding, thread_key, event_date, boost, emotional_intensity)
		VALUES (?, ?, ?, ?, ?, ?, ?)
	`, m.ID, m.Content, string(embJSON), m.ThreadKey, m.EventDate, m.Boost, m.EmotionalIntensity)
	if err != nil {
		return fmt.Errorf("save memory: %w", err)
	}
	return nil
}

func (s *SQLiteStore) DeleteMemory(ctx context.Context, id string) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM memories WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("delete memory: %w", err)
	}
	return nil
}

func (s *SQLiteStore) UpdateBoost(ctx context.Context, id string, delta float64) error {
	_, err := s.db.ExecContext(ctx, `UPDATE memories SET boost = boost + ? WHERE id = ?`, delta, id)
	if err != nil {
		return fmt.Errorf("update boost: %w", err)
	}
	return nil
}

// MemoryEntry is a simplified view of a memory for display.
type MemoryEntry struct {
	ID        string
	Content   string
	EventDate string
	ThreadKey string
}

// ListRecent returns the most recent n memories ordered by rowid descending.
func (s *SQLiteStore) ListRecent(ctx context.Context, limit int) ([]MemoryEntry, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, content, event_date, thread_key
		FROM memories
		ORDER BY rowid DESC
		LIMIT ?
	`, limit)
	if err != nil {
		return nil, fmt.Errorf("list recent: %w", err)
	}
	defer rows.Close()

	var entries []MemoryEntry
	for rows.Next() {
		var e MemoryEntry
		if err := rows.Scan(&e.ID, &e.Content, &e.EventDate, &e.ThreadKey); err != nil {
			return nil, fmt.Errorf("scan entry: %w", err)
		}
		entries = append(entries, e)
	}
	return entries, rows.Err()
}

// SearchContent returns memories whose content contains the query string.
func (s *SQLiteStore) SearchContent(ctx context.Context, query string, limit int) ([]MemoryEntry, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, content, event_date, thread_key
		FROM memories
		WHERE content LIKE '%' || ? || '%'
		ORDER BY rowid DESC
		LIMIT ?
	`, query, limit)
	if err != nil {
		return nil, fmt.Errorf("search content: %w", err)
	}
	defer rows.Close()

	var entries []MemoryEntry
	for rows.Next() {
		var e MemoryEntry
		if err := rows.Scan(&e.ID, &e.Content, &e.EventDate, &e.ThreadKey); err != nil {
			return nil, fmt.Errorf("scan entry: %w", err)
		}
		entries = append(entries, e)
	}
	return entries, rows.Err()
}

// TokenUsageRecord represents a single token usage entry.
type TokenUsageRecord struct {
	Timestamp        string
	Model            string
	Turn             int
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// SaveTokenUsage records token consumption for a single turn.
func (s *SQLiteStore) SaveTokenUsage(ctx context.Context, version, model string, turn, promptTokens, completionTokens int) error {
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO token_usage (version, model, turn, prompt_tokens, completion_tokens, total_tokens)
		VALUES (?, ?, ?, ?, ?, ?)
	`, version, model, turn, promptTokens, completionTokens, promptTokens+completionTokens)
	if err != nil {
		return fmt.Errorf("save token usage: %w", err)
	}
	return nil
}

// TokenUsageSummary returns aggregated token usage grouped by date, version and model.
func (s *SQLiteStore) TokenUsageSummary(ctx context.Context, limit int) ([]TokenUsageDailySummary, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT date(timestamp) as day, version, model,
			COUNT(*) as turns,
			SUM(prompt_tokens) as prompt_total,
			SUM(completion_tokens) as completion_total,
			SUM(total_tokens) as total
		FROM token_usage
		GROUP BY day, version, model
		ORDER BY day DESC
		LIMIT ?
	`, limit)
	if err != nil {
		return nil, fmt.Errorf("token usage summary: %w", err)
	}
	defer rows.Close()

	var summaries []TokenUsageDailySummary
	for rows.Next() {
		var s TokenUsageDailySummary
		if err := rows.Scan(&s.Date, &s.Version, &s.Model, &s.Turns, &s.PromptTokens, &s.CompletionTokens, &s.TotalTokens); err != nil {
			return nil, fmt.Errorf("scan summary: %w", err)
		}
		summaries = append(summaries, s)
	}
	return summaries, rows.Err()
}

// TokenUsageDailySummary holds aggregated token usage for a single day, version and model.
type TokenUsageDailySummary struct {
	Date             string
	Version          string
	Model            string
	Turns            int
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// MemoryDetail is a full view of a memory for the memory subcommand.
type MemoryDetail struct {
	ID                 string
	Content            string
	EventDate          string
	ThreadKey          string
	Boost              float64
	EmotionalIntensity float64
}

// ListMemories returns the most recent n memories with metadata.
func (s *SQLiteStore) ListMemories(ctx context.Context, limit int) ([]MemoryDetail, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, content, event_date, thread_key, boost, emotional_intensity
		FROM memories
		ORDER BY rowid DESC
		LIMIT ?
	`, limit)
	if err != nil {
		return nil, fmt.Errorf("list memories: %w", err)
	}
	defer rows.Close()

	var entries []MemoryDetail
	for rows.Next() {
		var e MemoryDetail
		if err := rows.Scan(&e.ID, &e.Content, &e.EventDate, &e.ThreadKey, &e.Boost, &e.EmotionalIntensity); err != nil {
			return nil, fmt.Errorf("scan memory detail: %w", err)
		}
		entries = append(entries, e)
	}
	return entries, rows.Err()
}

// SearchMemories returns memories whose content contains the query string, with metadata.
func (s *SQLiteStore) SearchMemories(ctx context.Context, query string, limit int) ([]MemoryDetail, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, content, event_date, thread_key, boost, emotional_intensity
		FROM memories
		WHERE content LIKE '%' || ? || '%'
		ORDER BY rowid DESC
		LIMIT ?
	`, query, limit)
	if err != nil {
		return nil, fmt.Errorf("search memories: %w", err)
	}
	defer rows.Close()

	var entries []MemoryDetail
	for rows.Next() {
		var e MemoryDetail
		if err := rows.Scan(&e.ID, &e.Content, &e.EventDate, &e.ThreadKey, &e.Boost, &e.EmotionalIntensity); err != nil {
			return nil, fmt.Errorf("scan memory detail: %w", err)
		}
		entries = append(entries, e)
	}
	return entries, rows.Err()
}

// CountMemories returns the total number of memories.
func (s *SQLiteStore) CountMemories(ctx context.Context) (int, error) {
	var count int
	err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM memories`).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("count memories: %w", err)
	}
	return count, nil
}

func (s *SQLiteStore) Close() error {
	return s.db.Close()
}
