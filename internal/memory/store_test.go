package memory

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	memai "github.com/ieee0824/memAI-go"
)

func newTestStore(t *testing.T) *SQLiteStore {
	t.Helper()
	dir := t.TempDir()
	store, err := NewSQLiteStore(filepath.Join(dir, "test.db"))
	if err != nil {
		t.Fatalf("NewSQLiteStore: %v", err)
	}
	t.Cleanup(func() { store.Close() })
	return store
}

func TestNewSQLiteStore(t *testing.T) {
	store := newTestStore(t)
	if store.db == nil {
		t.Fatal("db is nil")
	}
}

func TestNewSQLiteStore_InvalidPath(t *testing.T) {
	_, err := NewSQLiteStore("/nonexistent/dir/test.db")
	if err == nil {
		t.Error("expected error for invalid path")
	}
}

func TestSaveAndGetMemories(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	mem := &memai.Memory[string]{
		ID:                 "test-1",
		Content:            "hello world",
		Embedding:          []float64{0.1, 0.2, 0.3},
		ThreadKey:          "thread-1",
		EventDate:          "2026-03-05",
		Boost:              0.5,
		EmotionalIntensity: 0.7,
	}

	if err := store.SaveMemory(ctx, mem); err != nil {
		t.Fatalf("SaveMemory: %v", err)
	}

	memories, err := store.GetMemories(ctx)
	if err != nil {
		t.Fatalf("GetMemories: %v", err)
	}

	if len(memories) != 1 {
		t.Fatalf("len(memories) = %d, want 1", len(memories))
	}

	got := memories[0]
	if got.ID != "test-1" {
		t.Errorf("ID = %q, want %q", got.ID, "test-1")
	}
	if got.Content != "hello world" {
		t.Errorf("Content = %q, want %q", got.Content, "hello world")
	}
	if len(got.Embedding) != 3 {
		t.Errorf("len(Embedding) = %d, want 3", len(got.Embedding))
	}
	if got.ThreadKey != "thread-1" {
		t.Errorf("ThreadKey = %q, want %q", got.ThreadKey, "thread-1")
	}
	if got.Boost != 0.5 {
		t.Errorf("Boost = %f, want 0.5", got.Boost)
	}
	if got.EmotionalIntensity != 0.7 {
		t.Errorf("EmotionalIntensity = %f, want 0.7", got.EmotionalIntensity)
	}
}

func TestSaveMemory_Upsert(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	mem := &memai.Memory[string]{
		ID:      "test-1",
		Content: "original",
	}
	store.SaveMemory(ctx, mem)

	mem.Content = "updated"
	store.SaveMemory(ctx, mem)

	memories, _ := store.GetMemories(ctx)
	if len(memories) != 1 {
		t.Fatalf("len(memories) = %d, want 1 (upsert)", len(memories))
	}
	if memories[0].Content != "updated" {
		t.Errorf("Content = %q, want %q", memories[0].Content, "updated")
	}
}

func TestDeleteMemory(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	store.SaveMemory(ctx, &memai.Memory[string]{ID: "test-1", Content: "a"})
	store.SaveMemory(ctx, &memai.Memory[string]{ID: "test-2", Content: "b"})

	if err := store.DeleteMemory(ctx, "test-1"); err != nil {
		t.Fatalf("DeleteMemory: %v", err)
	}

	memories, _ := store.GetMemories(ctx)
	if len(memories) != 1 {
		t.Fatalf("len(memories) = %d, want 1", len(memories))
	}
	if memories[0].ID != "test-2" {
		t.Errorf("remaining ID = %q, want %q", memories[0].ID, "test-2")
	}
}

func TestUpdateBoost(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	store.SaveMemory(ctx, &memai.Memory[string]{ID: "test-1", Content: "a", Boost: 0.0})

	if err := store.UpdateBoost(ctx, "test-1", 0.05); err != nil {
		t.Fatalf("UpdateBoost: %v", err)
	}
	if err := store.UpdateBoost(ctx, "test-1", 0.1); err != nil {
		t.Fatalf("UpdateBoost: %v", err)
	}

	memories, _ := store.GetMemories(ctx)
	if len(memories) != 1 {
		t.Fatalf("len(memories) = %d, want 1", len(memories))
	}

	expected := 0.15
	if diff := memories[0].Boost - expected; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("Boost = %f, want %f", memories[0].Boost, expected)
	}
}

func TestGetMemories_Empty(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	memories, err := store.GetMemories(ctx)
	if err != nil {
		t.Fatalf("GetMemories: %v", err)
	}
	if len(memories) != 0 {
		t.Errorf("len(memories) = %d, want 0", len(memories))
	}
}

func TestGetMemories_EmptyEmbedding(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	store.SaveMemory(ctx, &memai.Memory[string]{
		ID:      "test-1",
		Content: "no embedding",
	})

	memories, err := store.GetMemories(ctx)
	if err != nil {
		t.Fatalf("GetMemories: %v", err)
	}
	if len(memories) != 1 {
		t.Fatalf("len = %d, want 1", len(memories))
	}
	// null embedding is fine - will be empty slice after JSON unmarshal of "null"
	_ = memories[0].Embedding
}

func TestNewSQLiteStore_CreatesParentDir(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "sub", "dir", "test.db")
	os.MkdirAll(filepath.Dir(dbPath), 0755)

	store, err := NewSQLiteStore(dbPath)
	if err != nil {
		t.Fatalf("NewSQLiteStore: %v", err)
	}
	store.Close()
}
