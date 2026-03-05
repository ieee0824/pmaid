package skills

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Skill represents a loaded skill file.
type Skill struct {
	Name    string // filename without extension
	Content string
	Source  string // "global" or "project"
}

// Load reads skill files from the given directories.
// Later directories override earlier ones for files with the same name.
// Supported extensions: .md, .txt
func Load(dirs ...string) []Skill {
	byName := make(map[string]Skill)

	for _, dir := range dirs {
		source := "global"
		if !isHomeDir(dir) {
			source = "project"
		}
		loadDir(dir, source, byName)
	}

	// Sort by name for deterministic order
	skills := make([]Skill, 0, len(byName))
	for _, s := range byName {
		skills = append(skills, s)
	}
	sort.Slice(skills, func(i, j int) bool {
		return skills[i].Name < skills[j].Name
	})

	return skills
}

func loadDir(dir, source string, byName map[string]Skill) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return // directory doesn't exist — that's fine
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		ext := filepath.Ext(entry.Name())
		if ext != ".md" && ext != ".txt" {
			continue
		}

		data, err := os.ReadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			continue
		}

		name := strings.TrimSuffix(entry.Name(), ext)
		content := strings.TrimSpace(string(data))
		if content == "" {
			continue
		}

		byName[name] = Skill{
			Name:    name,
			Content: content,
			Source:  source,
		}
	}
}

func isHomeDir(dir string) bool {
	home, err := os.UserHomeDir()
	if err != nil {
		return false
	}
	return strings.HasPrefix(dir, home)
}

// GlobalDir returns ~/.pmaid/skills
func GlobalDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".pmaid", "skills")
}

// ProjectDir returns <contextDir>/.pmaid/skills
func ProjectDir(contextDir string) string {
	return filepath.Join(contextDir, ".pmaid", "skills")
}

// AgentDir returns <contextDir>/.agent/skills
func AgentDir(contextDir string) string {
	return filepath.Join(contextDir, ".agent", "skills")
}

// FormatForPrompt renders loaded skills as a system prompt section.
func FormatForPrompt(skills []Skill) string {
	if len(skills) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("## Custom Skills\n")
	for _, s := range skills {
		sb.WriteString(fmt.Sprintf("\n### %s\n%s\n", s.Name, s.Content))
	}
	return sb.String()
}
