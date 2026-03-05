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
		return nil, fmt.Errorf("create table: %w", err)
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

func (s *SQLiteStore) Close() error {
	return s.db.Close()
}
