package skills

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoad_Empty(t *testing.T) {
	dir := t.TempDir()
	result := Load(dir)
	if len(result) != 0 {
		t.Errorf("expected 0 skills, got %d", len(result))
	}
}

func TestLoad_NonExistentDir(t *testing.T) {
	result := Load("/nonexistent/path")
	if len(result) != 0 {
		t.Errorf("expected 0 skills, got %d", len(result))
	}
}

func TestLoad_BasicFiles(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "coding.md"), []byte("Always use gofmt"), 0644)
	os.WriteFile(filepath.Join(dir, "review.txt"), []byte("Check for errors"), 0644)

	result := Load(dir)
	if len(result) != 2 {
		t.Fatalf("expected 2 skills, got %d", len(result))
	}

	// Sorted by name
	if result[0].Name != "coding" {
		t.Errorf("first skill name = %q, want coding", result[0].Name)
	}
	if result[0].Content != "Always use gofmt" {
		t.Errorf("first skill content = %q", result[0].Content)
	}
	if result[1].Name != "review" {
		t.Errorf("second skill name = %q, want review", result[1].Name)
	}
}

func TestLoad_IgnoresUnsupportedExtensions(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "skill.md"), []byte("valid"), 0644)
	os.WriteFile(filepath.Join(dir, "data.json"), []byte(`{"key":"val"}`), 0644)
	os.WriteFile(filepath.Join(dir, "script.sh"), []byte("#!/bin/bash"), 0644)

	result := Load(dir)
	if len(result) != 1 {
		t.Errorf("expected 1 skill, got %d", len(result))
	}
}

func TestLoad_IgnoresEmptyFiles(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "empty.md"), []byte(""), 0644)
	os.WriteFile(filepath.Join(dir, "whitespace.md"), []byte("   \n  "), 0644)

	result := Load(dir)
	if len(result) != 0 {
		t.Errorf("expected 0 skills, got %d", len(result))
	}
}

func TestLoad_IgnoresDirectories(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "subdir.md"), 0755)
	os.WriteFile(filepath.Join(dir, "real.md"), []byte("content"), 0644)

	result := Load(dir)
	if len(result) != 1 {
		t.Errorf("expected 1 skill, got %d", len(result))
	}
}

func TestLoad_ProjectOverridesGlobal(t *testing.T) {
	globalDir := t.TempDir()
	projectDir := t.TempDir()

	os.WriteFile(filepath.Join(globalDir, "style.md"), []byte("global style"), 0644)
	os.WriteFile(filepath.Join(projectDir, "style.md"), []byte("project style"), 0644)
	os.WriteFile(filepath.Join(globalDir, "global-only.md"), []byte("only in global"), 0644)

	result := Load(globalDir, projectDir)
	if len(result) != 2 {
		t.Fatalf("expected 2 skills, got %d", len(result))
	}

	// Find "style" — should be project version
	for _, s := range result {
		if s.Name == "style" {
			if s.Content != "project style" {
				t.Errorf("style content = %q, want project version", s.Content)
			}
			return
		}
	}
	t.Error("style skill not found")
}

func TestFormatForPrompt_Empty(t *testing.T) {
	result := FormatForPrompt(nil)
	if result != "" {
		t.Errorf("expected empty string, got %q", result)
	}
}

func TestFormatForPrompt_WithSkills(t *testing.T) {
	skills := []Skill{
		{Name: "coding", Content: "Use gofmt"},
		{Name: "review", Content: "Check errors"},
	}
	result := FormatForPrompt(skills)

	if !strings.Contains(result, "## Custom Skills") {
		t.Error("expected header")
	}
	if !strings.Contains(result, "### coding") {
		t.Error("expected coding skill header")
	}
	if !strings.Contains(result, "Use gofmt") {
		t.Error("expected coding skill content")
	}
	if !strings.Contains(result, "### review") {
		t.Error("expected review skill header")
	}
}
