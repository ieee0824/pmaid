package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type FileWrite struct {
	baseDir string
}

func NewFileWrite(baseDir string) *FileWrite {
	return &FileWrite{baseDir: baseDir}
}

func (f *FileWrite) Name() string        { return "write_file" }
func (f *FileWrite) Description() string  { return "Write content to a file, creating parent directories if needed" }
func (f *FileWrite) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "File path to write (relative to context directory or absolute)",
			},
			"content": map[string]interface{}{
				"type":        "string",
				"description": "Content to write to the file",
			},
		},
		"required": []string{"path", "content"},
	}
}

func (f *FileWrite) Execute(_ interface{}, args string) (string, error) {
	var params struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("parse args: %w", err)
	}

	path := params.Path
	if !filepath.IsAbs(path) {
		path = filepath.Join(f.baseDir, path)
	}

	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return "", fmt.Errorf("create directories: %w", err)
	}

	if err := os.WriteFile(path, []byte(params.Content), 0644); err != nil {
		return "", fmt.Errorf("write file: %w", err)
	}

	return fmt.Sprintf("Successfully wrote %d bytes to %s", len(params.Content), params.Path), nil
}
