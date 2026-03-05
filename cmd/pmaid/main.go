package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	memai "github.com/ieee0824/memAI-go"
	"github.com/ieee0824/pmaid/internal/agent"
	oaillm "github.com/ieee0824/pmaid/internal/llm/openai"
	"github.com/ieee0824/pmaid/internal/memory"
	"github.com/ieee0824/pmaid/internal/tools"
)

func main() {
	query := flag.String("q", "", "Direct query (non-interactive mode)")
	contextDir := flag.String("context", ".", "Context directory for file operations")
	model := flag.String("model", "gpt-4o", "LLM model to use")
	flag.Parse()

	absContext, err := filepath.Abs(*contextDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error resolving context dir: %v\n", err)
		os.Exit(1)
	}

	// Memory path
	memPath := os.Getenv("PMAID_MEMORY_PATH")
	if memPath == "" {
		home, _ := os.UserHomeDir()
		memPath = filepath.Join(home, ".pmaid", "memory")
	}
	if err := os.MkdirAll(memPath, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating memory dir: %v\n", err)
		os.Exit(1)
	}

	// SQLite store
	store, err := memory.NewSQLiteStore(filepath.Join(memPath, "memories.db"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening memory store: %v\n", err)
		os.Exit(1)
	}
	defer store.Close()

	// Embedder
	embedder := memory.NewTFIDFEmbedder(512)

	// STM
	stm := memai.NewSTM(memai.STMConfig{})

	// LTM
	ltm := memai.NewLTM[string](store, embedder.EmbedFunc(), memai.LTMConfig{
		TopK: 5,
	})

	// LLM client
	apiKey := os.Getenv("OPENAI_API_KEY")
	llmClient := oaillm.New(*model, apiKey)

	// Tools
	toolRegistry := tools.NewRegistry(
		tools.NewFileRead(absContext),
		tools.NewFileWrite(absContext),
		tools.NewExec(absContext),
	)

	// Agent
	ag := agent.New(agent.Config{
		LLMClient:  llmClient,
		STM:        stm,
		LTM:        ltm,
		Store:      store,
		Tools:      toolRegistry,
		Embedder:   embedder.EmbedFunc(),
		ContextDir: absContext,
	})

	ctx := context.Background()

	if *query != "" {
		result, err := ag.Run(ctx, *query)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(result)
		return
	}

	// Interactive mode
	fmt.Println("pmaid - Programming AI Assistant with Memory")
	fmt.Println("Type your message (Ctrl+D to exit)")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer

	for {
		fmt.Print("you> ")
		if !scanner.Scan() {
			break
		}
		input := scanner.Text()
		if input == "" {
			continue
		}

		result, err := ag.Run(ctx, input)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}
		fmt.Printf("\npmaid> %s\n\n", result)
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
	}
}
