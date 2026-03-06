package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"

	memai "github.com/ieee0824/memAI-go"
	"github.com/ieee0824/pmaid/internal/agent"
	"github.com/ieee0824/pmaid/internal/config"
	"github.com/ieee0824/pmaid/internal/llm"
	googlm "github.com/ieee0824/pmaid/internal/llm/google"
	oaillm "github.com/ieee0824/pmaid/internal/llm/openai"
	"github.com/ieee0824/pmaid/internal/logger"
	"github.com/ieee0824/pmaid/internal/memory"
	"github.com/ieee0824/pmaid/internal/skills"
	"github.com/ieee0824/pmaid/internal/spinner"
	"github.com/ieee0824/pmaid/internal/tools"
	"github.com/ieee0824/pmaid/internal/ui"
)

// version is set via -ldflags at build time.
// Falls back to Go module version (for go install).
var version = ""

func getVersion() string {
	if version != "" {
		return version
	}
	if info, ok := debug.ReadBuildInfo(); ok && info.Main.Version != "" && info.Main.Version != "(devel)" {
		return info.Main.Version
	}
	return "dev"
}

func main() {
	styles := ui.NewStyles()

	// Handle subcommands before flag parsing
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "history":
			runHistory(os.Args[2:], styles)
			return
		case "memory":
			runMemory(os.Args[2:], styles)
			return
		case "usage":
			runUsage(os.Args[2:], styles)
			return
		case "--version", "version":
			fmt.Println(getVersion())
			return
		case "completion":
			printBashCompletion()
			return
		}
	}

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
			fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error during setup:"), setupErr)
			os.Exit(1)
		}
		fmt.Println()
	} else {
		var loadErr error
		cfg, loadErr = config.Load(cfgPath)
		if loadErr != nil {
			fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error loading config:"), loadErr)
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
		fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error resolving context dir:"), err)
		os.Exit(1)
	}

	// Memory path
	memPath := cfg.ResolveMemoryPath()
	if err := os.MkdirAll(memPath, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error creating memory dir:"), err)
		os.Exit(1)
	}

	// Logger
	log, err := logger.New(filepath.Join(memPath, "logs"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error creating logger:"), err)
		os.Exit(1)
	}
	defer log.Close()

	// SQLite store
	store, err := memory.NewSQLiteStore(filepath.Join(memPath, "memories.db"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error opening memory store:"), err)
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
	var llmOpts []oaillm.Option
	if cfg.LLM.BaseURL != "" {
		llmOpts = append(llmOpts, oaillm.WithBaseURL(cfg.LLM.BaseURL))
	}
	llmClient := oaillm.New(cfg.LLM.Model, apiKey, llmOpts...)

	// Light LLM client (optional, for context compression etc.)
	var lightClient llm.Client
	if cfg.HasLightLLM() {
		lightKey := cfg.ResolveLightAPIKey()
		switch cfg.LightLLM.Provider {
		case "google":
			lightClient = googlm.New(cfg.LightLLM.Model, lightKey, cfg.LightLLM.BaseURL)
		case "local", "openai":
			var lightOpts []oaillm.Option
			if cfg.LightLLM.BaseURL != "" {
				lightOpts = append(lightOpts, oaillm.WithBaseURL(cfg.LightLLM.BaseURL))
			}
			lightClient = oaillm.New(cfg.LightLLM.Model, lightKey, lightOpts...)
		default:
			var lightOpts []oaillm.Option
			if cfg.LightLLM.BaseURL != "" {
				lightOpts = append(lightOpts, oaillm.WithBaseURL(cfg.LightLLM.BaseURL))
			}
			lightClient = oaillm.New(cfg.LightLLM.Model, lightKey, lightOpts...)
		}
	}

	// Show model info on startup
	if cfg.LLM.Model != "" {
		fmt.Fprintf(os.Stderr, "LLM(model): %s\n", cfg.LLM.Model)
	}
	if cfg.HasLightLLM() {
		fmt.Fprintf(os.Stderr, "Light LLM(provider=%s, model): %s\n", cfg.LightLLM.Provider, cfg.LightLLM.Model)
	}

	// Plan holder (shared between tools and agent)
	planHolder := &tools.PlanHolder{}

	// Tools
	toolRegistry := tools.NewRegistry(
		tools.NewFileRead(absContext),
		tools.NewFileWrite(absContext),
		tools.NewFileEdit(absContext),
		tools.NewExec(absContext),
		tools.NewWebFetch(),
		tools.NewCreatePlan(planHolder),
		tools.NewUpdatePlanStep(planHolder),
		tools.NewShowPlan(planHolder),
	)

	// Load skills from ~/.pmaid/skills, ./.pmaid/skills, ./.agent/skills
	loadedSkills := skills.Load(skills.GlobalDir(), skills.ProjectDir(absContext), skills.AgentDir(absContext))
	skillsCtx := skills.FormatForPrompt(loadedSkills)

	// Agent
	ag := agent.New(agent.Config{
		LLMClient:         llmClient,
		LightClient:       lightClient,
		STM:               stm,
		LTM:               ltm,
		Store:             store,
		Tools:             toolRegistry,
		PlanHolder:        planHolder,
		Embedder:          embedder.EmbedFunc(),
		ContextDir:        absContext,
		Name:              cfg.Name,
		Version:           getVersion(),
		Model:             cfg.LLM.Model,
		SkillsContext:     skillsCtx,
		Logger:            log,
		MaxToolIterations: cfg.Agent.MaxToolIterations,
		MaxContextChars:   cfg.Agent.MaxContextChars,
	})

	agentName := ag.Name()

	// Spinner manager for pausing/resuming during confirmation
	var activeSpinner *spinner.Spinner

	confirmScanner := bufio.NewScanner(os.Stdin)
	confirmScanner.Buffer(make([]byte, 1024), 1024)

	ag.SetOnConfirm(func(desc string) bool {
		// Pause spinner while waiting for user input
		if activeSpinner != nil {
			activeSpinner.Stop()
		}
		fmt.Fprintf(os.Stderr, "\n%s %s\n", styles.Warn("⚠"), desc)
		fmt.Fprint(os.Stderr, "実行しますか？ [y/n]: ")
		if !confirmScanner.Scan() {
			activeSpinner = spinner.New(os.Stderr, "考え中...")
			ag.SetOnStatus(activeSpinner.SetMessage)
			activeSpinner.Start()
			return false
		}
		answer := strings.TrimSpace(strings.ToLower(confirmScanner.Text()))
		approved := answer == "y" || answer == "yes" || answer == "はい"
		if !approved {
			fmt.Fprintln(os.Stderr, styles.Info("却下しました。"))
		}
		// Resume spinner
		activeSpinner = spinner.New(os.Stderr, "考え中...")
		ag.SetOnStatus(activeSpinner.SetMessage)
		activeSpinner.Start()
		return approved
	})

	ctx := context.Background()

	if *query != "" {
		activeSpinner = spinner.New(os.Stderr, "考え中...")
		ag.SetOnStatus(activeSpinner.SetMessage)
		activeSpinner.Start()
		result, err := ag.Run(ctx, *query)
		activeSpinner.Stop()
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error:"), err)
			os.Exit(1)
		}
		fmt.Println(result)
		return
	}

	// Interactive mode
	fmt.Printf("%s - Programming AI Assistant with Memory\n", styles.Banner(agentName))
	fmt.Println("Type your message (empty line to send, Ctrl+D to exit)")
	if !styles.Enabled {
		fmt.Println("(hint) set PMAID_FORCE_COLOR=1 to force color; set NO_COLOR=1 to disable")
	}
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer

	for {
		fmt.Print(styles.PromptMe("you> "))
		var lines []string
		eof := true
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				eof = false
				break
			}
			lines = append(lines, line)
			fmt.Print(styles.PromptMe("...> "))
		}
		if len(lines) == 0 {
			if eof {
				break
			}
			continue
		}
		input := strings.Join(lines, "\n")

		activeSpinner = spinner.New(os.Stderr, "考え中...")
		ag.SetOnStatus(activeSpinner.SetMessage)
		activeSpinner.Start()
		result, err := ag.Run(ctx, input)
		activeSpinner.Stop()
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error:"), err)
			continue
		}
		fmt.Printf("\n%s %s\n\n", styles.PromptAI(agentName+">"), result)

		// Plan approval flow
		if ag.HasPendingPlan() {
			handlePlanApproval(ag, scanner, ctx, &activeSpinner, agentName, styles)
		}

		if eof {
			break
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error reading input:"), err)
	}
}

func runUsage(args []string, styles ui.Styles) {
	fs := flag.NewFlagSet("usage", flag.ExitOnError)
	limit := fs.Int("n", 30, "Number of summary rows to show")
	configPath := fs.String("config", "", "Config file path")
	fs.Parse(args)

	cfgPath := *configPath
	if cfgPath == "" {
		cfgPath = config.DefaultConfigPath()
	}
	cfg, err := config.Load(cfgPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error loading config:"), err)
		os.Exit(1)
	}

	memPath := cfg.ResolveMemoryPath()
	store, err := memory.NewSQLiteStore(filepath.Join(memPath, "memories.db"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error opening memory store:"), err)
		os.Exit(1)
	}
	defer store.Close()

	ctx := context.Background()
	summaries, err := store.TokenUsageSummary(ctx, *limit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error:"), err)
		os.Exit(1)
	}

	if len(summaries) == 0 {
		fmt.Println("トークン使用履歴がありません。")
		return
	}

	fmt.Printf("%-12s %-10s %-20s %6s %10s %10s %10s %10s\n",
		"Date", "Version", "Model", "Turns", "Prompt", "Completion", "Total", "Avg/Turn")
	fmt.Println(strings.Repeat("─", 95))
	var grandTotal, grandTurns int
	for _, s := range summaries {
		avg := 0
		if s.Turns > 0 {
			avg = s.TotalTokens / s.Turns
		}
		fmt.Printf("%-12s %-10s %-20s %6d %10d %10d %10d %10d\n",
			s.Date, s.Version, s.Model, s.Turns, s.PromptTokens, s.CompletionTokens, s.TotalTokens, avg)
		grandTotal += s.TotalTokens
		grandTurns += s.Turns
	}
	fmt.Println(strings.Repeat("─", 95))
	grandAvg := 0
	if grandTurns > 0 {
		grandAvg = grandTotal / grandTurns
	}
	fmt.Printf("%73s %10d %10d\n", "合計:", grandTotal, grandAvg)
}

func runMemory(args []string, styles ui.Styles) {
	fs := flag.NewFlagSet("memory", flag.ExitOnError)
	limit := fs.Int("n", 20, "Number of entries to show")
	configPath := fs.String("config", "", "Config file path")
	fs.Parse(args)

	cfgPath := *configPath
	if cfgPath == "" {
		cfgPath = config.DefaultConfigPath()
	}
	cfg, err := config.Load(cfgPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error loading config:"), err)
		os.Exit(1)
	}

	memPath := cfg.ResolveMemoryPath()
	store, err := memory.NewSQLiteStore(filepath.Join(memPath, "memories.db"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error opening memory store:"), err)
		os.Exit(1)
	}
	defer store.Close()

	ctx := context.Background()

	total, err := store.CountMemories(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error:"), err)
		os.Exit(1)
	}

	searchQuery := fs.Arg(0)
	var entries []memory.MemoryDetail
	if searchQuery != "" {
		entries, err = store.SearchMemories(ctx, searchQuery, *limit)
	} else {
		entries, err = store.ListMemories(ctx, *limit)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error:"), err)
		os.Exit(1)
	}

	if len(entries) == 0 {
		fmt.Println("メモリがありません。")
		return
	}

	fmt.Printf("メモリ総数: %d件\n\n", total)

	for i := len(entries) - 1; i >= 0; i-- {
		e := entries[i]
		fmt.Printf("%s %s %s\n", styles.Info("───"), e.EventDate, styles.Info("───"))
		fmt.Printf("  ID: %s\n", e.ID[:8])
		if e.Boost != 0 {
			fmt.Printf("  boost: %+.1f", e.Boost)
			if e.EmotionalIntensity > 0 {
				fmt.Printf("  emotion: %.2f", e.EmotionalIntensity)
			}
			fmt.Println()
		} else if e.EmotionalIntensity > 0 {
			fmt.Printf("  emotion: %.2f\n", e.EmotionalIntensity)
		}
		// Show content summary
		content := e.Content
		parts := strings.SplitN(content, "\nAssistant: ", 2)
		if len(parts) == 2 {
			userMsg := strings.TrimPrefix(parts[0], "User: ")
			fmt.Printf("  %s   %s\n", styles.PromptMe("you>"), truncateStr(userMsg, 100))
			fmt.Printf("  %s %s\n", styles.PromptAI("pmaid>"), truncateStr(parts[1], 150))
		} else {
			fmt.Printf("  %s\n", truncateStr(content, 200))
		}
		fmt.Println()
	}

	if searchQuery != "" {
		fmt.Printf("(%d件表示, 検索: %q)\n", len(entries), searchQuery)
	} else {
		fmt.Printf("(%d件表示 / %d件中)\n", len(entries), total)
	}
}

func runHistory(args []string, styles ui.Styles) {
	fs := flag.NewFlagSet("history", flag.ExitOnError)
	limit := fs.Int("n", 20, "Number of entries to show")
	configPath := fs.String("config", "", "Config file path")
	fs.Parse(args)

	cfgPath := *configPath
	if cfgPath == "" {
		cfgPath = config.DefaultConfigPath()
	}
	cfg, err := config.Load(cfgPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error loading config:"), err)
		os.Exit(1)
	}

	memPath := cfg.ResolveMemoryPath()
	store, err := memory.NewSQLiteStore(filepath.Join(memPath, "memories.db"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error opening memory store:"), err)
		os.Exit(1)
	}
	defer store.Close()

	ctx := context.Background()
	searchQuery := fs.Arg(0)

	var entries []memory.MemoryEntry
	if searchQuery != "" {
		entries, err = store.SearchContent(ctx, searchQuery, *limit)
	} else {
		entries, err = store.ListRecent(ctx, *limit)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error:"), err)
		os.Exit(1)
	}

	if len(entries) == 0 {
		fmt.Println("会話履歴がありません。")
		return
	}

	// Display in chronological order (entries are newest-first)
	for i := len(entries) - 1; i >= 0; i-- {
		e := entries[i]
		fmt.Printf("%s %s %s\n", styles.Info("───"), e.EventDate, styles.Info("───"))
		// Parse "User: ...\nAssistant: ..." format
		parts := strings.SplitN(e.Content, "\nAssistant: ", 2)
		if len(parts) == 2 {
			userMsg := strings.TrimPrefix(parts[0], "User: ")
			fmt.Printf("  %s   %s\n", styles.PromptMe("you>"), userMsg)
			fmt.Printf("  %s %s\n", styles.PromptAI("pmaid>"), truncateStr(parts[1], 200))
		} else {
			fmt.Printf("  %s\n", truncateStr(e.Content, 200))
		}
		fmt.Println()
	}
	fmt.Printf("(%d件表示)\n", len(entries))
}

func printBashCompletion() {
	fmt.Print(`_pmaid() {
    local cur prev commands flags
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    commands="history memory usage version completion"
    flags="-q -context -model -config"

    if [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "${commands} ${flags}" -- "${cur}") )
        return 0
    fi

    case "${prev}" in
        -q|-context|-config)
            COMPREPLY=()
            ;;
        -model)
            COMPREPLY=()
            ;;
        history|memory|usage)
            COMPREPLY=( $(compgen -W "-n -config" -- "${cur}") )
            ;;
        *)
            COMPREPLY=( $(compgen -W "${flags}" -- "${cur}") )
            ;;
    esac
}
complete -F _pmaid pmaid
`)
}

func truncateStr(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func handlePlanApproval(ag *agent.Agent, scanner *bufio.Scanner, ctx context.Context, spRef **spinner.Spinner, agentName string, styles ui.Styles) {
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
			*spRef = spinner.New(os.Stderr, "実行中...")
			ag.SetOnStatus((*spRef).SetMessage)
			(*spRef).Start()
			result, err := ag.Run(ctx, "Plan approved. Please execute the plan step by step.")
			(*spRef).Stop()
			if err != nil {
				fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error:"), err)
				return
			}
			fmt.Printf("\n%s %s\n\n", styles.PromptAI(agentName+">"), result)
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
			*spRef = spinner.New(os.Stderr, "修正中...")
			ag.SetOnStatus((*spRef).SetMessage)
			(*spRef).Start()
			result, err := ag.Run(ctx, fmt.Sprintf("The previous plan was rejected. Please revise it with this feedback: %s", feedback))
			(*spRef).Stop()
			if err != nil {
				fmt.Fprintf(os.Stderr, "%s %v\n", styles.Error("Error:"), err)
				return
			}
			fmt.Printf("\n%s %s\n\n", styles.PromptAI(agentName+">"), result)
			// Loop again for new plan approval if a new plan was created
			if !ag.HasPendingPlan() {
				return
			}

		default:
			fmt.Println("y/n/e で回答してください")
		}
	}
}
