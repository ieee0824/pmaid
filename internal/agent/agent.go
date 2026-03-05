package agent

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	memai "github.com/ieee0824/memAI-go"
	"github.com/ieee0824/pmaid/internal/llm"
	"github.com/ieee0824/pmaid/internal/tools"
)

const maxToolIterations = 10

type Agent struct {
	llmClient  llm.Client
	stm        *memai.STM
	ltm        *memai.LTM[string]
	store      MemoryStoreWithBoost
	toolReg    *tools.Registry
	planHolder *tools.PlanHolder
	embedder   EmbedFunc
	contextDir string
	threadKey  string
	turn       int
	history    []llm.Message
}

type EmbedFunc func(ctx context.Context, text string) ([]float64, error)

type MemoryStoreWithBoost interface {
	SaveMemory(ctx context.Context, mem *memai.Memory[string]) error
	UpdateBoost(ctx context.Context, id string, delta float64) error
}

type Config struct {
	LLMClient  llm.Client
	STM        *memai.STM
	LTM        *memai.LTM[string]
	Store      MemoryStoreWithBoost
	Tools      *tools.Registry
	PlanHolder *tools.PlanHolder
	Embedder   EmbedFunc
	ContextDir string
}

func New(cfg Config) *Agent {
	return &Agent{
		llmClient:  cfg.LLMClient,
		stm:        cfg.STM,
		ltm:        cfg.LTM,
		store:      cfg.Store,
		toolReg:    cfg.Tools,
		planHolder: cfg.PlanHolder,
		embedder:   cfg.Embedder,
		contextDir: cfg.ContextDir,
		threadKey:  uuid.New().String(),
		history:    []llm.Message{},
	}
}

// HasPendingPlan returns true if there is a plan waiting for user approval.
func (a *Agent) HasPendingPlan() bool {
	if a.planHolder == nil || a.planHolder.Current == nil {
		return false
	}
	return !a.planHolder.Current.IsApproved()
}

// ApprovePlan approves the current plan.
func (a *Agent) ApprovePlan() {
	if a.planHolder != nil && a.planHolder.Current != nil {
		a.planHolder.Current.Approve()
	}
}

// RejectPlan clears the current plan.
func (a *Agent) RejectPlan() {
	if a.planHolder != nil {
		a.planHolder.Current = nil
	}
}

// CurrentPlan returns the current plan string representation, or empty.
func (a *Agent) CurrentPlan() string {
	if a.planHolder == nil || a.planHolder.Current == nil {
		return ""
	}
	return a.planHolder.Current.String()
}

func (a *Agent) Run(ctx context.Context, userInput string) (string, error) {
	a.turn++

	// Emotion analysis
	emotion := memai.AnalyzeEmotion(userInput, memai.LangJapanese)
	if emotion.Primary == memai.EmotionNeutral {
		emotion = memai.AnalyzeEmotion(userInput, memai.LangEnglish)
	}

	// Feedback detection and boost
	delta := memai.DetectFeedback(userInput)
	if delta != 0 {
		results, _ := a.ltm.Search(ctx, memai.SearchQuery{
			Query:     userInput,
			ThreadKey: a.threadKey,
			QueryDate: time.Now().Format("2006-01-02"),
		})
		for _, r := range results {
			a.store.UpdateBoost(ctx, r.Memory.ID, delta)
		}
	}

	// LTM search for relevant memories
	var memoryContext string
	results, err := a.ltm.Search(ctx, memai.SearchQuery{
		Query:              userInput,
		ThreadKey:          a.threadKey,
		QueryDate:          time.Now().Format("2006-01-02"),
		EmotionalIntensity: emotion.Intensity,
	})
	if err == nil && len(results) > 0 {
		var parts []string
		for _, r := range results {
			parts = append(parts, fmt.Sprintf("- %s (relevance: %.2f)", r.Memory.Content, r.Score))
		}
		memoryContext = "## Relevant Memories\n" + strings.Join(parts, "\n")
	}

	// STM update
	a.stm.Update(a.turn, userInput, emotion)

	// Build plan context
	var planContext string
	if a.planHolder != nil && a.planHolder.Current != nil {
		p := a.planHolder.Current
		if p.IsApproved() && !p.IsComplete() {
			planContext = "## Active Plan (approved)\n" + p.String()
		}
	}

	// Build system prompt
	systemPrompt := buildSystemPrompt(a.contextDir, memoryContext, planContext)

	// Build messages for LLM
	messages := []llm.Message{
		{Role: llm.RoleSystem, Content: systemPrompt},
	}
	messages = append(messages, a.history...)
	messages = append(messages, llm.Message{Role: llm.RoleUser, Content: userInput})

	// Tool-calling loop
	toolDefs := a.toolReg.Definitions()
	var finalResponse string

	for i := 0; i < maxToolIterations; i++ {
		resp, err := a.llmClient.Chat(ctx, messages, toolDefs)
		if err != nil {
			return "", fmt.Errorf("llm chat: %w", err)
		}

		if len(resp.Message.ToolCalls) == 0 {
			finalResponse = resp.Message.Content
			break
		}

		// Append assistant message with tool calls
		messages = append(messages, resp.Message)

		// Execute each tool call
		for _, tc := range resp.Message.ToolCalls {
			tool, ok := a.toolReg.Get(tc.Name)
			if !ok {
				messages = append(messages, llm.Message{
					Role:       llm.RoleTool,
					Content:    fmt.Sprintf("Error: unknown tool '%s'", tc.Name),
					ToolCallID: tc.ID,
				})
				continue
			}

			// Check if plan is required but not approved for destructive tools
			if a.isPlanRequiredTool(tc.Name) && a.hasPendingUnapprovedPlan() {
				messages = append(messages, llm.Message{
					Role:       llm.RoleTool,
					Content:    "Error: plan must be approved by user before executing this action. Wait for user approval.",
					ToolCallID: tc.ID,
				})
				continue
			}

			result, err := tool.Execute(ctx, tc.Arguments)
			if err != nil {
				result = fmt.Sprintf("Error: %s", err)
			}

			messages = append(messages, llm.Message{
				Role:       llm.RoleTool,
				Content:    result,
				ToolCallID: tc.ID,
			})

			// If a plan was just created, stop the tool loop to let user review
			if tc.Name == "create_plan" && a.HasPendingPlan() {
				finalResponse = result
				goto done
			}
		}
	}

done:
	// Update conversation history (keep last 20 messages to avoid unbounded growth)
	a.history = append(a.history,
		llm.Message{Role: llm.RoleUser, Content: userInput},
		llm.Message{Role: llm.RoleAssistant, Content: finalResponse},
	)
	if len(a.history) > 40 {
		a.history = a.history[len(a.history)-40:]
	}

	// Save to LTM
	a.saveToLTM(ctx, userInput, finalResponse, emotion)

	return finalResponse, nil
}

func (a *Agent) isPlanRequiredTool(name string) bool {
	switch name {
	case "write_file", "execute_command":
		return true
	}
	return false
}

func (a *Agent) hasPendingUnapprovedPlan() bool {
	if a.planHolder == nil || a.planHolder.Current == nil {
		return false
	}
	p := a.planHolder.Current
	return !p.IsApproved() && !p.IsComplete()
}

func (a *Agent) saveToLTM(ctx context.Context, input, response string, emotion *memai.EmotionalState) {
	content := fmt.Sprintf("User: %s\nAssistant: %s", input, response)
	embedding, err := a.embedder(ctx, content)
	if err != nil {
		return
	}

	mem := &memai.Memory[string]{
		ID:                 uuid.New().String(),
		Content:            content,
		Embedding:          embedding,
		ThreadKey:          a.threadKey,
		EventDate:          time.Now().Format("2006-01-02"),
		EmotionalIntensity: emotion.Intensity,
	}
	a.store.SaveMemory(ctx, mem)
}

func buildSystemPrompt(contextDir, memoryContext, planContext string) string {
	var sb strings.Builder
	sb.WriteString(`You are pmaid, a programming AI assistant with memory.
You help users with software engineering tasks including writing code, debugging, file operations, and running commands.

## Guidelines
- Be concise and direct
- When asked to modify files, use the write_file tool
- When asked to read files, use the read_file tool
- When asked to run commands, use the execute_command tool
- Always explain what you're doing before using tools

## Planning
- For large tasks (multi-file changes, refactoring, new features with multiple components), ALWAYS create a plan first using create_plan
- A plan breaks the task into clear, ordered steps
- After creating a plan, STOP and wait for user approval — do NOT execute any write_file or execute_command until approved
- Once approved, follow the plan step by step, updating each step's status with update_plan_step
- Mark steps as "in_progress" when starting and "completed" when done
- For simple tasks (single file edit, quick question), skip planning and execute directly

## Typo Handling
- User input may contain typos, misspellings, or grammatical errors in both Japanese and English
- Automatically interpret the user's intent despite any typos and proceed with the corrected interpretation
- Do NOT ask for clarification on obvious typos — silently correct and execute
- For file paths or code identifiers with typos, infer the correct name from context (e.g. existing files, common patterns)
- If a correction significantly changes the meaning, briefly mention what you interpreted (e.g. "「〇〇」として処理します")
- Examples: "ふぁいるよんで" → read file, "comit" → commit, "tset" → test
`)

	if contextDir != "" {
		sb.WriteString(fmt.Sprintf("\n## Context\nWorking directory: %s\n", contextDir))
	}

	if planContext != "" {
		sb.WriteString("\n" + planContext + "\n")
	}

	if memoryContext != "" {
		sb.WriteString("\n" + memoryContext + "\n")
	}

	return sb.String()
}
