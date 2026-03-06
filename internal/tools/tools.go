package tools

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/ieee0824/pmaid/internal/llm"
)

// safePath resolves and validates that the given path stays within baseDir.
// It prevents path traversal attacks using "..", absolute paths, and symlinks.
func safePath(baseDir, path string) (string, error) {
	if !filepath.IsAbs(path) {
		path = filepath.Join(baseDir, path)
	}
	path = filepath.Clean(path)

	// If the file itself exists, resolve it fully
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		// File doesn't exist yet — walk up to find the nearest existing ancestor
		resolved = resolveWithAncestor(path)
	}

	absBase, err := filepath.EvalSymlinks(baseDir)
	if err != nil {
		absBase = filepath.Clean(baseDir)
	}

	if !strings.HasPrefix(resolved, absBase+string(filepath.Separator)) && resolved != absBase {
		return "", fmt.Errorf("path %q is outside allowed directory %q", path, baseDir)
	}
	return path, nil
}

// resolveWithAncestor walks up from path until it finds an existing directory,
// resolves symlinks on that ancestor, then re-appends the remaining segments.
func resolveWithAncestor(path string) string {
	parts := []string{filepath.Base(path)}
	dir := filepath.Dir(path)
	for {
		resolved, err := filepath.EvalSymlinks(dir)
		if err == nil {
			// Found an existing ancestor — rejoin remaining parts
			for i := len(parts) - 1; i >= 0; i-- {
				resolved = filepath.Join(resolved, parts[i])
			}
			return resolved
		}
		parts = append(parts, filepath.Base(dir))
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return path
}

type Tool interface {
	Name() string
	Description() string
	Parameters() map[string]interface{}
	Execute(ctx interface{}, args string) (string, error)
}

type Registry struct {
	tools map[string]Tool
}

func NewRegistry(tt ...Tool) *Registry {
	r := &Registry{tools: make(map[string]Tool)}
	for _, t := range tt {
		r.tools[t.Name()] = t
	}
	return r
}

// Get returns a tool by exact name, or falls back to fuzzy matching
// if no exact match is found.
func (r *Registry) Get(name string) (Tool, bool) {
	if t, ok := r.tools[name]; ok {
		return t, true
	}
	return r.fuzzyGet(name)
}

func (r *Registry) fuzzyGet(name string) (Tool, bool) {
	var bestTool Tool
	bestDist := -1
	for registered, t := range r.tools {
		d := editDistance(name, registered)
		maxLen := len(name)
		if len(registered) > maxLen {
			maxLen = len(registered)
		}
		threshold := maxLen / 3
		if threshold < 2 {
			threshold = 2
		}
		if d <= threshold && (bestDist < 0 || d < bestDist) {
			bestDist = d
			bestTool = t
		}
	}
	if bestTool != nil {
		return bestTool, true
	}
	return nil, false
}

func editDistance(a, b string) int {
	la, lb := len(a), len(b)
	if la == 0 {
		return lb
	}
	if lb == 0 {
		return la
	}
	prev := make([]int, lb+1)
	curr := make([]int, lb+1)
	for j := 0; j <= lb; j++ {
		prev[j] = j
	}
	for i := 1; i <= la; i++ {
		curr[0] = i
		for j := 1; j <= lb; j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			del := prev[j] + 1
			ins := curr[j-1] + 1
			sub := prev[j-1] + cost
			v := del
			if ins < v {
				v = ins
			}
			if sub < v {
				v = sub
			}
			curr[j] = v
		}
		prev, curr = curr, prev
	}
	return prev[lb]
}

func (r *Registry) Definitions() []llm.ToolDef {
	defs := make([]llm.ToolDef, 0, len(r.tools))
	for _, t := range r.tools {
		defs = append(defs, llm.ToolDef{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  t.Parameters(),
		})
	}
	return defs
}

// DefinitionsExcluding returns tool definitions excluding the named tools.
func (r *Registry) DefinitionsExcluding(exclude map[string]bool) []llm.ToolDef {
	defs := make([]llm.ToolDef, 0, len(r.tools))
	for _, t := range r.tools {
		if exclude[t.Name()] {
			continue
		}
		defs = append(defs, llm.ToolDef{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  t.Parameters(),
		})
	}
	return defs
}

func (r *Registry) List() []Tool {
	tt := make([]Tool, 0, len(r.tools))
	for _, t := range r.tools {
		tt = append(tt, t)
	}
	return tt
}
