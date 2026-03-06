package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// FileEdit implements a line-based patch tool that replaces specific line ranges
// in a file, avoiding the need to send the entire file content as arguments.
type FileEdit struct {
	baseDir string
}

func NewFileEdit(baseDir string) *FileEdit {
	return &FileEdit{baseDir: baseDir}
}

func (f *FileEdit) Name() string { return "edit_file" }
func (f *FileEdit) Description() string {
	return "Edit a file by replacing specific line ranges. More token-efficient than write_file for small changes to large files. Use read_file first to see line numbers."
}
func (f *FileEdit) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"path": map[string]interface{}{
				"type":        "string",
				"description": "File path to edit (relative to context directory or absolute)",
			},
			"patches": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"start_line": map[string]interface{}{
							"type":        "integer",
							"description": "First line to replace (1-based, inclusive)",
						},
						"end_line": map[string]interface{}{
							"type":        "integer",
							"description": "Last line to replace (1-based, inclusive). Use same as start_line to replace a single line. Use 0 to insert before start_line without deleting.",
						},
						"content": map[string]interface{}{
							"type":        "string",
							"description": "Replacement content (may contain newlines). Empty string to delete lines.",
						},
					},
					"required": []string{"start_line", "end_line", "content"},
				},
				"description": "List of patches to apply, each replacing a line range. Applied in reverse order to preserve line numbers.",
			},
		},
		"required": []string{"path", "patches"},
	}
}

type patch struct {
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
	Content   string `json:"content"`
}

func (f *FileEdit) Execute(_ interface{}, args string) (string, error) {
	var params struct {
		Path    string  `json:"path"`
		Patches []patch `json:"patches"`
	}
	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("parse args: %w", err)
	}
	if len(params.Patches) == 0 {
		return "", fmt.Errorf("no patches provided")
	}

	path, err := safePath(f.baseDir, params.Path)
	if err != nil {
		return "", err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read file: %w", err)
	}

	lines := strings.Split(string(data), "\n")

	// Sort patches by start_line descending so we can apply from bottom to top
	// without invalidating line numbers
	sorted := make([]patch, len(params.Patches))
	copy(sorted, params.Patches)
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].StartLine > sorted[i].StartLine {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	for _, p := range sorted {
		if p.StartLine < 1 || p.StartLine > len(lines)+1 {
			return "", fmt.Errorf("start_line %d out of range (file has %d lines)", p.StartLine, len(lines))
		}

		var newLines []string
		if p.Content != "" {
			newLines = strings.Split(p.Content, "\n")
		}

		if p.EndLine == 0 {
			// Insert mode: insert before start_line without deleting
			idx := p.StartLine - 1
			result := make([]string, 0, len(lines)+len(newLines))
			result = append(result, lines[:idx]...)
			result = append(result, newLines...)
			result = append(result, lines[idx:]...)
			lines = result
		} else {
			// Replace mode
			if p.EndLine < p.StartLine {
				return "", fmt.Errorf("end_line %d < start_line %d", p.EndLine, p.StartLine)
			}
			if p.EndLine > len(lines) {
				return "", fmt.Errorf("end_line %d out of range (file has %d lines)", p.EndLine, len(lines))
			}
			start := p.StartLine - 1
			end := p.EndLine
			result := make([]string, 0, len(lines)-end+start+len(newLines))
			result = append(result, lines[:start]...)
			result = append(result, newLines...)
			result = append(result, lines[end:]...)
			lines = result
		}
	}

	output := strings.Join(lines, "\n")
	if err := os.WriteFile(path, []byte(output), 0644); err != nil {
		return "", fmt.Errorf("write file: %w", err)
	}

	return fmt.Sprintf("Successfully edited %s (%d patches applied, %d lines total)", params.Path, len(params.Patches), len(lines)), nil
}
