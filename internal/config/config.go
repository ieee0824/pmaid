package config

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

type Config struct {
	Model      string `toml:"model"`
	ContextDir string `toml:"context_dir"`
	MemoryPath string `toml:"memory_path"`

	LLM    LLMConfig    `toml:"llm"`
	Memory MemoryConfig `toml:"memory"`
	STM    STMConfig    `toml:"stm"`
	LTM    LTMConfig    `toml:"ltm"`
}

type LLMConfig struct {
	Provider string `toml:"provider"`
	APIKey   string `toml:"api_key"`
	Model    string `toml:"model"`
}

type MemoryConfig struct {
	EmbeddingDim int `toml:"embedding_dim"`
}

type STMConfig struct {
	MaxItems            int     `toml:"max_items"`
	ActivationThreshold float64 `toml:"activation_threshold"`
	NormalDecayRate     float64 `toml:"normal_decay_rate"`
	EmotionalDecayRate  float64 `toml:"emotional_decay_rate"`
	RefreshBoost        float64 `toml:"refresh_boost"`
}

type LTMConfig struct {
	TopK                int     `toml:"top_k"`
	SimilarityThreshold float64 `toml:"similarity_threshold"`
	ThreadBoost         float64 `toml:"thread_boost"`
	DateBoost           float64 `toml:"date_boost"`
	EmotionalBoost      float64 `toml:"emotional_boost"`
}

func DefaultConfig() Config {
	home, _ := os.UserHomeDir()
	return Config{
		Model:      "gpt-4o",
		ContextDir: ".",
		MemoryPath: filepath.Join(home, ".pmaid", "memory"),
		LLM: LLMConfig{
			Provider: "openai",
			Model:    "gpt-4o",
		},
		Memory: MemoryConfig{
			EmbeddingDim: 512,
		},
		STM: STMConfig{
			MaxItems:            7,
			ActivationThreshold: 0.1,
			NormalDecayRate:     0.15,
			EmotionalDecayRate:  0.07,
			RefreshBoost:        0.3,
		},
		LTM: LTMConfig{
			TopK:                5,
			SimilarityThreshold: 0.3,
			ThreadBoost:         0.1,
			DateBoost:           0.15,
			EmotionalBoost:      0.12,
		},
	}
}

// DefaultConfigPath returns ~/.pmaid/config.toml
func DefaultConfigPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".pmaid", "config.toml")
}

// Load reads a config file and merges with defaults.
// Missing fields keep their default values.
func Load(path string) (Config, error) {
	cfg := DefaultConfig()

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return cfg, fmt.Errorf("read config: %w", err)
	}

	if err := toml.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("parse config: %w", err)
	}

	// Sync top-level model with llm.model if llm.model is not explicitly set
	if cfg.LLM.Model == "" || cfg.LLM.Model == "gpt-4o" {
		if cfg.Model != "gpt-4o" {
			cfg.LLM.Model = cfg.Model
		}
	}

	return cfg, nil
}

// ResolveAPIKey returns the API key from config, falling back to env var.
func (c *Config) ResolveAPIKey() string {
	if c.LLM.APIKey != "" {
		return c.LLM.APIKey
	}
	switch c.LLM.Provider {
	case "openai":
		return os.Getenv("OPENAI_API_KEY")
	default:
		return os.Getenv("OPENAI_API_KEY")
	}
}

// ResolveMemoryPath returns the memory path from config, falling back to env var.
func (c *Config) ResolveMemoryPath() string {
	if envPath := os.Getenv("PMAID_MEMORY_PATH"); envPath != "" {
		return envPath
	}
	return c.MemoryPath
}
