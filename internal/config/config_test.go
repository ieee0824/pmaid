package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Model != "gpt-4o" {
		t.Errorf("Model = %q, want %q", cfg.Model, "gpt-4o")
	}
	if cfg.LLM.Provider != "openai" {
		t.Errorf("LLM.Provider = %q, want %q", cfg.LLM.Provider, "openai")
	}
	if cfg.Memory.EmbeddingDim != 512 {
		t.Errorf("Memory.EmbeddingDim = %d, want 512", cfg.Memory.EmbeddingDim)
	}
	if cfg.STM.MaxItems != 7 {
		t.Errorf("STM.MaxItems = %d, want 7", cfg.STM.MaxItems)
	}
	if cfg.LTM.TopK != 5 {
		t.Errorf("LTM.TopK = %d, want 5", cfg.LTM.TopK)
	}
}

func TestLoad_NonExistentFile(t *testing.T) {
	cfg, err := Load("/nonexistent/path/config.toml")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	// Should return defaults
	if cfg.Model != "gpt-4o" {
		t.Errorf("Model = %q, want default", cfg.Model)
	}
}

func TestLoad_ValidConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.toml")

	content := `
model = "gpt-4o-mini"
context_dir = "/home/user/project"

[llm]
provider = "openai"
api_key = "sk-test-key"
model = "gpt-4o-mini"

[memory]
embedding_dim = 256

[stm]
max_items = 10
normal_decay_rate = 0.2

[ltm]
top_k = 3
similarity_threshold = 0.5
`
	os.WriteFile(path, []byte(content), 0644)

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Model != "gpt-4o-mini" {
		t.Errorf("Model = %q, want %q", cfg.Model, "gpt-4o-mini")
	}
	if cfg.ContextDir != "/home/user/project" {
		t.Errorf("ContextDir = %q", cfg.ContextDir)
	}
	if cfg.LLM.APIKey != "sk-test-key" {
		t.Errorf("LLM.APIKey = %q", cfg.LLM.APIKey)
	}
	if cfg.LLM.Model != "gpt-4o-mini" {
		t.Errorf("LLM.Model = %q", cfg.LLM.Model)
	}
	if cfg.Memory.EmbeddingDim != 256 {
		t.Errorf("Memory.EmbeddingDim = %d, want 256", cfg.Memory.EmbeddingDim)
	}
	if cfg.STM.MaxItems != 10 {
		t.Errorf("STM.MaxItems = %d, want 10", cfg.STM.MaxItems)
	}
	if cfg.STM.NormalDecayRate != 0.2 {
		t.Errorf("STM.NormalDecayRate = %f, want 0.2", cfg.STM.NormalDecayRate)
	}
	if cfg.LTM.TopK != 3 {
		t.Errorf("LTM.TopK = %d, want 3", cfg.LTM.TopK)
	}
	if cfg.LTM.SimilarityThreshold != 0.5 {
		t.Errorf("LTM.SimilarityThreshold = %f, want 0.5", cfg.LTM.SimilarityThreshold)
	}
}

func TestLoad_PartialConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.toml")

	// Only override model, keep everything else default
	os.WriteFile(path, []byte(`model = "gpt-4o-mini"`), 0644)

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Model != "gpt-4o-mini" {
		t.Errorf("Model = %q, want %q", cfg.Model, "gpt-4o-mini")
	}
	// Defaults should be preserved
	if cfg.STM.MaxItems != 7 {
		t.Errorf("STM.MaxItems = %d, want default 7", cfg.STM.MaxItems)
	}
}

func TestLoad_InvalidTOML(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.toml")

	os.WriteFile(path, []byte(`invalid = [[[`), 0644)

	_, err := Load(path)
	if err == nil {
		t.Error("expected error for invalid TOML")
	}
}

func TestResolveAPIKey(t *testing.T) {
	t.Run("from config", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.LLM.APIKey = "sk-from-config"
		if got := cfg.ResolveAPIKey(); got != "sk-from-config" {
			t.Errorf("ResolveAPIKey = %q, want %q", got, "sk-from-config")
		}
	})

	t.Run("from env", func(t *testing.T) {
		cfg := DefaultConfig()
		t.Setenv("OPENAI_API_KEY", "sk-from-env")
		if got := cfg.ResolveAPIKey(); got != "sk-from-env" {
			t.Errorf("ResolveAPIKey = %q, want %q", got, "sk-from-env")
		}
	})
}

func TestResolveMemoryPath(t *testing.T) {
	t.Run("from env overrides config", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.MemoryPath = "/from/config"
		t.Setenv("PMAID_MEMORY_PATH", "/from/env")
		if got := cfg.ResolveMemoryPath(); got != "/from/env" {
			t.Errorf("ResolveMemoryPath = %q, want %q", got, "/from/env")
		}
	})

	t.Run("from config when env unset", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.MemoryPath = "/from/config"
		t.Setenv("PMAID_MEMORY_PATH", "")
		if got := cfg.ResolveMemoryPath(); got != "/from/config" {
			t.Errorf("ResolveMemoryPath = %q, want %q", got, "/from/config")
		}
	})
}
