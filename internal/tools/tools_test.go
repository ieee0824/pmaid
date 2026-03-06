package tools

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRegistry(t *testing.T) {
	reg := NewRegistry(
		NewFileRead("/tmp"),
		NewFileWrite("/tmp"),
		NewExec("/tmp"),
	)

	t.Run("Get existing tool", func(t *testing.T) {
		tool, ok := reg.Get("read_file")
		if !ok {
			t.Fatal("read_file not found")
		}
		if tool.Name() != "read_file" {
			t.Errorf("Name() = %q, want %q", tool.Name(), "read_file")
		}
	})

	t.Run("Get non-existing tool", func(t *testing.T) {
		_, ok := reg.Get("nonexistent")
		if ok {
			t.Error("expected not found")
		}
	})

	t.Run("Fuzzy match typo", func(t *testing.T) {
		tests := []struct {
			input string
			want  string
		}{
			{"read_flie", "read_file"},
			{"writ_file", "write_file"},
			{"execute_comand", "execute_command"},
			{"raed_file", "read_file"},
		}
		for _, tt := range tests {
			tool, ok := reg.Get(tt.input)
			if !ok {
				t.Errorf("Get(%q) not found, want %q", tt.input, tt.want)
				continue
			}
			if tool.Name() != tt.want {
				t.Errorf("Get(%q) = %q, want %q", tt.input, tool.Name(), tt.want)
			}
		}
	})

	t.Run("Fuzzy no match for distant names", func(t *testing.T) {
		_, ok := reg.Get("totally_different")
		if ok {
			t.Error("expected not found for very different name")
		}
	})

	t.Run("Definitions", func(t *testing.T) {
		defs := reg.Definitions()
		if len(defs) != 3 {
			t.Errorf("len(defs) = %d, want 3", len(defs))
		}
		names := make(map[string]bool)
		for _, d := range defs {
			names[d.Name] = true
		}
		for _, name := range []string{"read_file", "write_file", "execute_command"} {
			if !names[name] {
				t.Errorf("missing tool definition: %s", name)
			}
		}
	})

	t.Run("List", func(t *testing.T) {
		list := reg.List()
		if len(list) != 3 {
			t.Errorf("len(list) = %d, want 3", len(list))
		}
	})
}

func TestFileRead_Execute(t *testing.T) {
	dir := t.TempDir()
	content := "hello from test"
	os.WriteFile(filepath.Join(dir, "test.txt"), []byte(content), 0644)

	fr := NewFileRead(dir)

	t.Run("read relative path", func(t *testing.T) {
		result, err := fr.Execute(nil, `{"path": "test.txt"}`)
		if err != nil {
			t.Fatalf("Execute: %v", err)
		}
		if result != content {
			t.Errorf("result = %q, want %q", result, content)
		}
	})

	t.Run("read absolute path", func(t *testing.T) {
		absPath := filepath.Join(dir, "test.txt")
		result, err := fr.Execute(nil, `{"path": "`+absPath+`"}`)
		if err != nil {
			t.Fatalf("Execute: %v", err)
		}
		if result != content {
			t.Errorf("result = %q, want %q", result, content)
		}
	})

	t.Run("file not found", func(t *testing.T) {
		_, err := fr.Execute(nil, `{"path": "nonexistent.txt"}`)
		if err == nil {
			t.Error("expected error for nonexistent file")
		}
	})

	t.Run("invalid args", func(t *testing.T) {
		_, err := fr.Execute(nil, `invalid json`)
		if err == nil {
			t.Error("expected error for invalid JSON")
		}
	})
}

func TestFileRead_Truncation(t *testing.T) {
	dir := t.TempDir()
	// Create a file larger than 100KB
	bigContent := strings.Repeat("x", 200*1024)
	os.WriteFile(filepath.Join(dir, "big.txt"), []byte(bigContent), 0644)

	fr := NewFileRead(dir)
	result, err := fr.Execute(nil, `{"path": "big.txt"}`)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.HasSuffix(result, "... (truncated)") {
		t.Error("expected truncation suffix")
	}
	if len(result) > 110*1024 {
		t.Errorf("result too large: %d bytes", len(result))
	}
}

func TestFileWrite_Execute(t *testing.T) {
	dir := t.TempDir()
	fw := NewFileWrite(dir)

	t.Run("write new file", func(t *testing.T) {
		result, err := fw.Execute(nil, `{"path": "out.txt", "content": "hello"}`)
		if err != nil {
			t.Fatalf("Execute: %v", err)
		}
		if !strings.Contains(result, "Successfully wrote") {
			t.Errorf("unexpected result: %s", result)
		}

		data, _ := os.ReadFile(filepath.Join(dir, "out.txt"))
		if string(data) != "hello" {
			t.Errorf("file content = %q, want %q", string(data), "hello")
		}
	})

	t.Run("write with subdirectory creation", func(t *testing.T) {
		_, err := fw.Execute(nil, `{"path": "sub/dir/file.txt", "content": "nested"}`)
		if err != nil {
			t.Fatalf("Execute: %v", err)
		}
		data, _ := os.ReadFile(filepath.Join(dir, "sub", "dir", "file.txt"))
		if string(data) != "nested" {
			t.Errorf("file content = %q, want %q", string(data), "nested")
		}
	})

	t.Run("invalid args", func(t *testing.T) {
		_, err := fw.Execute(nil, `bad json`)
		if err == nil {
			t.Error("expected error for invalid JSON")
		}
	})
}

func TestFileEdit_Execute(t *testing.T) {
	dir := t.TempDir()
	original := "line1\nline2\nline3\nline4\nline5\n"
	os.WriteFile(filepath.Join(dir, "test.go"), []byte(original), 0644)

	fe := NewFileEdit(dir)

	t.Run("replace single line", func(t *testing.T) {
		os.WriteFile(filepath.Join(dir, "a.txt"), []byte("aaa\nbbb\nccc\n"), 0644)
		result, err := fe.Execute(nil, `{"path":"a.txt","patches":[{"start_line":2,"end_line":2,"content":"BBB"}]}`)
		if err != nil {
			t.Fatalf("Execute: %v", err)
		}
		if !strings.Contains(result, "1 patches") {
			t.Errorf("unexpected result: %s", result)
		}
		data, _ := os.ReadFile(filepath.Join(dir, "a.txt"))
		if string(data) != "aaa\nBBB\nccc\n" {
			t.Errorf("content = %q", string(data))
		}
	})

	t.Run("replace multiple lines", func(t *testing.T) {
		os.WriteFile(filepath.Join(dir, "b.txt"), []byte("1\n2\n3\n4\n5\n"), 0644)
		result, err := fe.Execute(nil, `{"path":"b.txt","patches":[{"start_line":2,"end_line":4,"content":"replaced"}]}`)
		if err != nil {
			t.Fatalf("Execute: %v", err)
		}
		data, _ := os.ReadFile(filepath.Join(dir, "b.txt"))
		if string(data) != "1\nreplaced\n5\n" {
			t.Errorf("content = %q", string(data))
		}
		_ = result
	})

	t.Run("insert mode (end_line=0)", func(t *testing.T) {
		os.WriteFile(filepath.Join(dir, "c.txt"), []byte("a\nb\n"), 0644)
		_, err := fe.Execute(nil, `{"path":"c.txt","patches":[{"start_line":2,"end_line":0,"content":"inserted"}]}`)
		if err != nil {
			t.Fatalf("Execute: %v", err)
		}
		data, _ := os.ReadFile(filepath.Join(dir, "c.txt"))
		if string(data) != "a\ninserted\nb\n" {
			t.Errorf("content = %q", string(data))
		}
	})

	t.Run("delete lines (empty content)", func(t *testing.T) {
		os.WriteFile(filepath.Join(dir, "d.txt"), []byte("a\nb\nc\n"), 0644)
		_, err := fe.Execute(nil, `{"path":"d.txt","patches":[{"start_line":2,"end_line":2,"content":""}]}`)
		if err != nil {
			t.Fatalf("Execute: %v", err)
		}
		data, _ := os.ReadFile(filepath.Join(dir, "d.txt"))
		if string(data) != "a\nc\n" {
			t.Errorf("content = %q", string(data))
		}
	})

	t.Run("multiple patches applied bottom-up", func(t *testing.T) {
		os.WriteFile(filepath.Join(dir, "e.txt"), []byte("a\nb\nc\nd\ne\n"), 0644)
		_, err := fe.Execute(nil, `{"path":"e.txt","patches":[{"start_line":2,"end_line":2,"content":"B"},{"start_line":4,"end_line":4,"content":"D"}]}`)
		if err != nil {
			t.Fatalf("Execute: %v", err)
		}
		data, _ := os.ReadFile(filepath.Join(dir, "e.txt"))
		if string(data) != "a\nB\nc\nD\ne\n" {
			t.Errorf("content = %q", string(data))
		}
	})

	t.Run("out of range error", func(t *testing.T) {
		os.WriteFile(filepath.Join(dir, "f.txt"), []byte("a\nb\n"), 0644)
		_, err := fe.Execute(nil, `{"path":"f.txt","patches":[{"start_line":10,"end_line":10,"content":"x"}]}`)
		if err == nil {
			t.Error("expected error for out of range")
		}
	})

	t.Run("no patches error", func(t *testing.T) {
		_, err := fe.Execute(nil, `{"path":"f.txt","patches":[]}`)
		if err == nil {
			t.Error("expected error for no patches")
		}
	})
}

func TestDefinitionsExcluding(t *testing.T) {
	reg := NewRegistry(
		NewFileRead("/tmp"),
		NewFileWrite("/tmp"),
		NewExec("/tmp"),
	)

	exclude := map[string]bool{"execute_command": true}
	defs := reg.DefinitionsExcluding(exclude)
	if len(defs) != 2 {
		t.Errorf("len(defs) = %d, want 2", len(defs))
	}
	for _, d := range defs {
		if d.Name == "execute_command" {
			t.Error("execute_command should be excluded")
		}
	}
}

func TestExec_Execute(t *testing.T) {
	dir := t.TempDir()
	ex := NewExec(dir)

	t.Run("simple command", func(t *testing.T) {
		result, err := ex.Execute(nil, `{"command": "echo hello"}`)
		if err != nil {
			t.Fatalf("Execute: %v", err)
		}
		if strings.TrimSpace(result) != "hello" {
			t.Errorf("result = %q, want %q", strings.TrimSpace(result), "hello")
		}
	})

	t.Run("command with stderr", func(t *testing.T) {
		result, err := ex.Execute(nil, `{"command": "echo err >&2"}`)
		if err != nil {
			t.Fatalf("Execute: %v", err)
		}
		if !strings.Contains(result, "[stderr]") {
			t.Errorf("expected [stderr] in result: %s", result)
		}
	})

	t.Run("failing command", func(t *testing.T) {
		result, err := ex.Execute(nil, `{"command": "exit 1"}`)
		if err != nil {
			t.Fatalf("Execute should not return error (captures it): %v", err)
		}
		if !strings.Contains(result, "[exit error") {
			t.Errorf("expected exit error in result: %s", result)
		}
	})

	t.Run("custom workdir", func(t *testing.T) {
		result, err := ex.Execute(nil, `{"command": "pwd", "workdir": "/tmp"}`)
		if err != nil {
			t.Fatalf("Execute: %v", err)
		}
		// /tmp may resolve to /private/tmp on macOS
		if !strings.Contains(result, "tmp") {
			t.Errorf("expected /tmp in result: %s", result)
		}
	})

	t.Run("invalid args", func(t *testing.T) {
		_, err := ex.Execute(nil, `bad`)
		if err == nil {
			t.Error("expected error for invalid JSON")
		}
	})
}
