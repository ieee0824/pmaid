package config

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Exists checks if the config file exists at the given path.
func Exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// InteractiveSetup prompts the user to create a config file interactively.
// Returns the created Config and nil error, or error if setup fails.
func InteractiveSetup(path string, r io.Reader, w io.Writer) (Config, error) {
	cfg := DefaultConfig()
	scanner := bufio.NewScanner(r)

	fmt.Fprintln(w, "=== pmaid 初期設定 ===")
	fmt.Fprintln(w)

	// LLM Provider
	fmt.Fprintf(w, "LLMプロバイダー [openai]: ")
	if scanner.Scan() {
		if v := strings.TrimSpace(scanner.Text()); v != "" {
			cfg.LLM.Provider = v
		}
	}

	// Model
	fmt.Fprintf(w, "モデル名 [%s]: ", cfg.LLM.Model)
	if scanner.Scan() {
		if v := strings.TrimSpace(scanner.Text()); v != "" {
			cfg.LLM.Model = v
			cfg.Model = v
		}
	}

	// API Key
	envKey := ""
	switch cfg.LLM.Provider {
	case "openai":
		envKey = os.Getenv("OPENAI_API_KEY")
	}
	if envKey != "" {
		fmt.Fprintf(w, "APIキー (環境変数から検出済み。設定ファイルにも保存する場合は入力) [スキップ]: ")
	} else {
		fmt.Fprintf(w, "APIキー (環境変数 OPENAI_API_KEY でも設定可): ")
	}
	if scanner.Scan() {
		if v := strings.TrimSpace(scanner.Text()); v != "" {
			cfg.LLM.APIKey = v
		}
	}

	// Memory path
	fmt.Fprintf(w, "メモリ保存先 [%s]: ", cfg.MemoryPath)
	if scanner.Scan() {
		if v := strings.TrimSpace(scanner.Text()); v != "" {
			cfg.MemoryPath = v
		}
	}

	fmt.Fprintln(w)

	// Write config file
	if err := WriteConfig(path, cfg); err != nil {
		return cfg, err
	}

	fmt.Fprintf(w, "設定を %s に保存しました\n", path)
	return cfg, scanner.Err()
}

// WriteConfig writes a Config to the given path as TOML.
func WriteConfig(path string, cfg Config) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("create config dir: %w", err)
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("model = %q\n", cfg.Model))
	sb.WriteString(fmt.Sprintf("context_dir = %q\n", cfg.ContextDir))
	sb.WriteString(fmt.Sprintf("memory_path = %q\n", cfg.MemoryPath))

	sb.WriteString("\n[llm]\n")
	sb.WriteString(fmt.Sprintf("provider = %q\n", cfg.LLM.Provider))
	if cfg.LLM.APIKey != "" {
		sb.WriteString(fmt.Sprintf("api_key = %q\n", cfg.LLM.APIKey))
	}
	sb.WriteString(fmt.Sprintf("model = %q\n", cfg.LLM.Model))

	sb.WriteString("\n[memory]\n")
	sb.WriteString(fmt.Sprintf("embedding_dim = %d\n", cfg.Memory.EmbeddingDim))

	sb.WriteString("\n[stm]\n")
	sb.WriteString(fmt.Sprintf("max_items = %d\n", cfg.STM.MaxItems))
	sb.WriteString(fmt.Sprintf("activation_threshold = %g\n", cfg.STM.ActivationThreshold))
	sb.WriteString(fmt.Sprintf("normal_decay_rate = %g\n", cfg.STM.NormalDecayRate))
	sb.WriteString(fmt.Sprintf("emotional_decay_rate = %g\n", cfg.STM.EmotionalDecayRate))
	sb.WriteString(fmt.Sprintf("refresh_boost = %g\n", cfg.STM.RefreshBoost))

	sb.WriteString("\n[ltm]\n")
	sb.WriteString(fmt.Sprintf("top_k = %d\n", cfg.LTM.TopK))
	sb.WriteString(fmt.Sprintf("similarity_threshold = %g\n", cfg.LTM.SimilarityThreshold))
	sb.WriteString(fmt.Sprintf("thread_boost = %g\n", cfg.LTM.ThreadBoost))
	sb.WriteString(fmt.Sprintf("date_boost = %g\n", cfg.LTM.DateBoost))
	sb.WriteString(fmt.Sprintf("emotional_boost = %g\n", cfg.LTM.EmotionalBoost))

	if err := os.WriteFile(path, []byte(sb.String()), 0600); err != nil {
		return fmt.Errorf("write config: %w", err)
	}
	return nil
}
