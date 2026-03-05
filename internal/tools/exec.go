package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"time"
)

type Exec struct {
	baseDir string
	timeout time.Duration
}

func NewExec(baseDir string) *Exec {
	return &Exec{baseDir: baseDir, timeout: 30 * time.Second}
}

func (e *Exec) Name() string        { return "execute_command" }
func (e *Exec) Description() string  { return "Execute a shell command and return stdout/stderr" }
func (e *Exec) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"command": map[string]interface{}{
				"type":        "string",
				"description": "Shell command to execute",
			},
			"workdir": map[string]interface{}{
				"type":        "string",
				"description": "Working directory (optional, defaults to context directory)",
			},
		},
		"required": []string{"command"},
	}
}

func (e *Exec) Execute(_ interface{}, args string) (string, error) {
	var params struct {
		Command string `json:"command"`
		Workdir string `json:"workdir"`
	}
	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("parse args: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), e.timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", params.Command)
	workdir := params.Workdir
	if workdir == "" {
		workdir = e.baseDir
	}
	cmd.Dir = workdir

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	output := stdout.String()
	if stderr.Len() > 0 {
		output += "\n[stderr]\n" + stderr.String()
	}

	if err != nil {
		output += fmt.Sprintf("\n[exit error: %s]", err)
	}

	return output, nil
}
