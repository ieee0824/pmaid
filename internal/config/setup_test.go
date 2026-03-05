package config

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExists(t *testing.T) {
	dir := t.TempDir()

	t.Run("file exists", func(t *testing.T) {
		path := filepath.Join(dir, "exists.toml")
		os.WriteFile(path, []byte(""), 0644)
		if !Exists(path) {
			t.Error("expected true for existing file")
		}
	})

	t.Run("file does not exist", func(t *testing.T) {
		if Exists(filepath.Join(dir, "nope.toml")) {
			t.Error("expected false for non-existing file")
		}
	})
}

func TestInteractiveSetup_Defaults(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.toml")

	// All empty input -> use defaults
	input := strings.NewReader("\n\n\n\n")
	var output bytes.Buffer

	cfg, err := InteractiveSetup(path, input, &output)
	if err != nil {
		t.Fatalf("InteractiveSetup: %v", err)
	}

	if cfg.LLM.Provider != "openai" {
		t.Errorf("Provider = %q, want openai", cfg.LLM.Provider)
	}
	if cfg.LLM.Model != "gpt-4o" {
		t.Errorf("Model = %q, want gpt-4o", cfg.LLM.Model)
	}

	// File should be created
	if !Exists(path) {
		t.Error("config file not created")
	}

	// Should be loadable
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.LLM.Model != "gpt-4o" {
		t.Errorf("loaded Model = %q", loaded.LLM.Model)
	}
}

func TestInteractiveSetup_CustomValues(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.toml")

	input := strings.NewReader("openai\ngpt-4o-mini\nsk-testkey\n/tmp/pmaid-mem\n")
	var output bytes.Buffer

	cfg, err := InteractiveSetup(path, input, &output)
	if err != nil {
		t.Fatalf("InteractiveSetup: %v", err)
	}

	if cfg.LLM.Model != "gpt-4o-mini" {
		t.Errorf("Model = %q, want gpt-4o-mini", cfg.LLM.Model)
	}
	if cfg.LLM.APIKey != "sk-testkey" {
		t.Errorf("APIKey = %q, want sk-testkey", cfg.LLM.APIKey)
	}
	if cfg.MemoryPath != "/tmp/pmaid-mem" {
		t.Errorf("MemoryPath = %q", cfg.MemoryPath)
	}

	// Verify file contains custom values
	data, _ := os.ReadFile(path)
	content := string(data)
	if !strings.Contains(content, "gpt-4o-mini") {
		t.Error("config file should contain model name")
	}
	if !strings.Contains(content, "sk-testkey") {
		t.Error("config file should contain api key")
	}
}

func TestInteractiveSetup_CreatesParentDirs(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sub", "dir", "config.toml")

	input := strings.NewReader("\n\n\n\n")
	var output bytes.Buffer

	_, err := InteractiveSetup(path, input, &output)
	if err != nil {
		t.Fatalf("InteractiveSetup: %v", err)
	}
	if !Exists(path) {
		t.Error("config file not created in nested directory")
	}
}

func TestInteractiveSetup_FilePermissions(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.toml")

	input := strings.NewReader("\n\nsk-secret\n\n")
	var output bytes.Buffer

	InteractiveSetup(path, input, &output)

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	// File should be 0600 (owner read/write only) since it may contain API keys
	perm := info.Mode().Perm()
	if perm != 0600 {
		t.Errorf("permissions = %o, want 0600", perm)
	}
}

func TestWriteConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.toml")

	cfg := DefaultConfig()
	cfg.LLM.APIKey = "sk-test"

	if err := WriteConfig(path, cfg); err != nil {
		t.Fatalf("WriteConfig: %v", err)
	}

	// Verify it's valid TOML by loading it back
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.LLM.APIKey != "sk-test" {
		t.Errorf("APIKey = %q, want sk-test", loaded.LLM.APIKey)
	}
	if loaded.STM.MaxItems != 7 {
		t.Errorf("STM.MaxItems = %d, want 7", loaded.STM.MaxItems)
	}
}
