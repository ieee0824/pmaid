package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
)

type FileRead struct {
	baseDir string
}

func NewFileRead(baseDir string) *FileRead {
	return &FileRead{baseDir: baseDir}
}

func (f *FileRead) Name() string        { return "read_file" }
func (f *FileRead) Description() string  { return "Read the contents of a file" }
func (f *FileRead) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "File path to read (relative to context directory or absolute)",
			},
		},
		"required": []string{"path"},
	}
}

func (f *FileRead) Execute(_ interface{}, args string) (string, error) {
	var params struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("parse args: %w", err)
	}

	path, err := safePath(f.baseDir, params.Path)
	if err != nil {
		return "", err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read file: %w", err)
	}

	const maxSize = 100 * 1024
	if len(data) > maxSize {
		return string(data[:maxSize]) + "\n... (truncated)", nil
	}
	return string(data), nil
}

// Ensure FileRead implements Tool via context.Context
func (f *FileRead) ExecuteCtx(ctx context.Context, args string) (string, error) {
	return f.Execute(ctx, args)
}
