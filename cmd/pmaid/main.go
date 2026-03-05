package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	memai "github.com/ieee0824/memAI-go"
	"github.com/ieee0824/pmaid/internal/agent"
	"github.com/ieee0824/pmaid/internal/config"
	oaillm "github.com/ieee0824/pmaid/internal/llm/openai"
	"github.com/ieee0824/pmaid/internal/memory"
	"github.com/ieee0824/pmaid/internal/spinner"
	"github.com/ieee0824/pmaid/internal/tools"
)

func main() {
	query := flag.String("q", "", "Direct query (non-interactive mode)")
	contextDir := flag.String("context", "", "Context directory for file operations")
	model := flag.String("model", "", "LLM model to use")
	configPath := flag.String("config", "", "Config file path (default: ~/.pmaid/config.toml)")
	flag.Parse()

	// Load config
	cfgPath := *configPath
	if cfgPath == "" {
		cfgPath = config.DefaultConfigPath()
	}

	var cfg config.Config
	if !config.Exists(cfgPath) {
		fmt.Printf("設定ファイルが見つかりません: %s\n\n", cfgPath)
		var setupErr error
		cfg, setupErr = config.InteractiveSetup(cfgPath, os.Stdin, os.Stdout)
		if setupErr != nil {
			fmt.Fprintf(os.Stderr, "Error during setup: %v\n", setupErr)
			os.Exit(1)
		}
		fmt.Println()
	} else {
		var loadErr error
		cfg, loadErr = config.Load(cfgPath)
		if loadErr != nil {
			fmt.Fprintf(os.Stderr, "Error loading config: %v\n", loadErr)
			os.Exit(1)
		}
	}

	// CLI flags override config
	if *model != "" {
		cfg.Model = *model
		cfg.LLM.Model = *model
	}
	if *contextDir != "" {
		cfg.ContextDir = *contextDir
	}

	absContext, err := filepath.Abs(cfg.ContextDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error resolving context dir: %v\n", err)
		os.Exit(1)
	}

	// Memory path
	memPath := cfg.ResolveMemoryPath()
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
	embedder := memory.NewTFIDFEmbedder(cfg.Memory.EmbeddingDim)

	// STM
	stm := memai.NewSTM(memai.STMConfig{
		MaxItems:            cfg.STM.MaxItems,
		ActivationThreshold: cfg.STM.ActivationThreshold,
		NormalDecayRate:     cfg.STM.NormalDecayRate,
		EmotionalDecayRate:  cfg.STM.EmotionalDecayRate,
		RefreshBoost:        cfg.STM.RefreshBoost,
	})

	// LTM
	ltm := memai.NewLTM[string](store, embedder.EmbedFunc(), memai.LTMConfig{
		TopK:                cfg.LTM.TopK,
		SimilarityThreshold: cfg.LTM.SimilarityThreshold,
		ThreadBoost:         cfg.LTM.ThreadBoost,
		DateBoost:           cfg.LTM.DateBoost,
		EmotionalBoost:      cfg.LTM.EmotionalBoost,
	})

	// LLM client
	apiKey := cfg.ResolveAPIKey()
	llmClient := oaillm.New(cfg.LLM.Model, apiKey)

	// Plan holder (shared between tools and agent)
	planHolder := &tools.PlanHolder{}

	// Tools
	toolRegistry := tools.NewRegistry(
		tools.NewFileRead(absContext),
		tools.NewFileWrite(absContext),
		tools.NewExec(absContext),
		tools.NewCreatePlan(planHolder),
		tools.NewUpdatePlanStep(planHolder),
		tools.NewShowPlan(planHolder),
	)

	// Agent
	ag := agent.New(agent.Config{
		LLMClient:  llmClient,
		STM:        stm,
		LTM:        ltm,
		Store:      store,
		Tools:      toolRegistry,
		PlanHolder: planHolder,
		Embedder:   embedder.EmbedFunc(),
		ContextDir: absContext,
	})

	ctx := context.Background()

	if *query != "" {
		sp := spinner.New(os.Stderr, "考え中...")
		sp.Start()
		result, err := ag.Run(ctx, *query)
		sp.Stop()
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

		sp := spinner.New(os.Stderr, "考え中...")
		sp.Start()
		result, err := ag.Run(ctx, input)
		sp.Stop()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}
		fmt.Printf("\npmaid> %s\n\n", result)

		// Plan approval flow
		if ag.HasPendingPlan() {
			handlePlanApproval(ag, scanner, ctx)
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
	}
}

func handlePlanApproval(ag *agent.Agent, scanner *bufio.Scanner, ctx context.Context) {
	for {
		fmt.Print("\nプランを承認しますか？ [y: 承認 / n: 却下 / e: 修正依頼]: ")
		if !scanner.Scan() {
			return
		}
		response := strings.TrimSpace(strings.ToLower(scanner.Text()))

		switch response {
		case "y", "yes", "はい":
			ag.ApprovePlan()
			fmt.Print("\nプランを承認しました。実行を開始します。\n\n")
			sp := spinner.New(os.Stderr, "実行中...")
			sp.Start()
			result, err := ag.Run(ctx, "Plan approved. Please execute the plan step by step.")
			sp.Stop()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				return
			}
			fmt.Printf("\npmaid> %s\n\n", result)
			return

		case "n", "no", "いいえ":
			ag.RejectPlan()
			fmt.Print("\nプランを却下しました。\n\n")
			return

		case "e", "edit", "修正":
			fmt.Print("修正内容を入力してください: ")
			if !scanner.Scan() {
				return
			}
			feedback := scanner.Text()
			ag.RejectPlan()
			sp := spinner.New(os.Stderr, "修正中...")
			sp.Start()
			result, err := ag.Run(ctx, fmt.Sprintf("The previous plan was rejected. Please revise it with this feedback: %s", feedback))
			sp.Stop()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				return
			}
			fmt.Printf("\npmaid> %s\n\n", result)
			// Loop again for new plan approval if a new plan was created
			if !ag.HasPendingPlan() {
				return
			}

		default:
			fmt.Println("y/n/e で回答してください")
		}
	}
}
