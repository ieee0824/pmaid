package agent

import (
	"context"
	"fmt"
	"strings"
	"testing"

	memai "github.com/ieee0824/memAI-go"
	"github.com/ieee0824/pmaid/internal/llm"
	"github.com/ieee0824/pmaid/internal/plan"
	"github.com/ieee0824/pmaid/internal/tools"
)

// mockLLMClient is a test double for llm.Client
type mockLLMClient struct {
	responses []*llm.Response
	callCount int
	messages  [][]llm.Message // recorded messages per call
}

func (m *mockLLMClient) Chat(_ context.Context, messages []llm.Message, _ []llm.ToolDef) (*llm.Response, error) {
	m.messages = append(m.messages, messages)
	if m.callCount >= len(m.responses) {
		return nil, fmt.Errorf("no more mock responses")
	}
	resp := m.responses[m.callCount]
	m.callCount++
	return resp, nil
}

// mockStore implements MemoryStoreWithBoost
type mockStore struct {
	memories []memai.Memory[string]
	boosts   map[string]float64
}

func newMockStore() *mockStore {
	return &mockStore{boosts: make(map[string]float64)}
}

func (s *mockStore) SaveMemory(_ context.Context, mem *memai.Memory[string]) error {
	s.memories = append(s.memories, *mem)
	return nil
}

func (s *mockStore) DeleteMemory(_ context.Context, id string) error {
	return nil
}

func (s *mockStore) UpdateBoost(_ context.Context, id string, delta float64) error {
	s.boosts[id] += delta
	return nil
}

func (s *mockStore) SaveTokenUsage(_ context.Context, _, _ string, _, _, _ int) error {
	return nil
}

func (s *mockStore) GetMemories(_ context.Context) ([]memai.Memory[string], error) {
	return s.memories, nil
}

func dummyEmbedder(_ context.Context, _ string) ([]float64, error) {
	return make([]float64, 64), nil
}

func TestAgent_SimpleResponse(t *testing.T) {
	mock := &mockLLMClient{
		responses: []*llm.Response{
			{Message: llm.Message{Role: llm.RoleAssistant, Content: "Hello! How can I help?"}},
		},
	}
	store := newMockStore()

	ag := New(Config{
		LLMClient:  mock,
		STM:        memai.NewSTM(memai.STMConfig{}),
		LTM:        memai.NewLTM[string](store, dummyEmbedder, memai.LTMConfig{}),
		Store:      store,
		Tools:      tools.NewRegistry(),
		Embedder:   dummyEmbedder,
		ContextDir: "/tmp",
	})

	result, err := ag.Run(context.Background(), "Hello")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result != "Hello! How can I help?" {
		t.Errorf("result = %q, want %q", result, "Hello! How can I help?")
	}

	// Verify LLM was called once
	if mock.callCount != 1 {
		t.Errorf("callCount = %d, want 1", mock.callCount)
	}

	// Verify system prompt is included
	msgs := mock.messages[0]
	if msgs[0].Role != llm.RoleSystem {
		t.Errorf("first message role = %q, want system", msgs[0].Role)
	}

	// Verify memory was saved
	if len(store.memories) != 1 {
		t.Errorf("saved memories = %d, want 1", len(store.memories))
	}
}

func TestAgent_ToolCalling(t *testing.T) {
	mock := &mockLLMClient{
		responses: []*llm.Response{
			// First response: LLM requests a tool call
			{Message: llm.Message{
				Role:    llm.RoleAssistant,
				Content: "",
				ToolCalls: []llm.ToolCall{
					{ID: "call-1", Name: "execute_command", Arguments: `{"command": "echo test"}`},
				},
			}},
			// Second response: LLM returns final answer
			{Message: llm.Message{Role: llm.RoleAssistant, Content: "The command output: test"}},
		},
	}
	store := newMockStore()

	ag := New(Config{
		LLMClient:  mock,
		STM:        memai.NewSTM(memai.STMConfig{}),
		LTM:        memai.NewLTM[string](store, dummyEmbedder, memai.LTMConfig{}),
		Store:      store,
		Tools:      tools.NewRegistry(tools.NewExec("/tmp")),
		Embedder:   dummyEmbedder,
		ContextDir: "/tmp",
	})

	result, err := ag.Run(context.Background(), "run echo test")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result != "The command output: test" {
		t.Errorf("result = %q", result)
	}
	if mock.callCount != 2 {
		t.Errorf("callCount = %d, want 2", mock.callCount)
	}

	// Second call should include tool result
	msgs := mock.messages[1]
	var hasToolMsg bool
	for _, m := range msgs {
		if m.Role == llm.RoleTool {
			hasToolMsg = true
			break
		}
	}
	if !hasToolMsg {
		t.Error("expected tool message in second LLM call")
	}
}

func TestAgent_UnknownTool(t *testing.T) {
	mock := &mockLLMClient{
		responses: []*llm.Response{
			{Message: llm.Message{
				Role: llm.RoleAssistant,
				ToolCalls: []llm.ToolCall{
					{ID: "call-1", Name: "nonexistent_tool", Arguments: `{}`},
				},
			}},
			{Message: llm.Message{Role: llm.RoleAssistant, Content: "Sorry, I tried an unknown tool."}},
		},
	}
	store := newMockStore()

	ag := New(Config{
		LLMClient:  mock,
		STM:        memai.NewSTM(memai.STMConfig{}),
		LTM:        memai.NewLTM[string](store, dummyEmbedder, memai.LTMConfig{}),
		Store:      store,
		Tools:      tools.NewRegistry(),
		Embedder:   dummyEmbedder,
		ContextDir: "/tmp",
	})

	result, err := ag.Run(context.Background(), "do something")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result != "Sorry, I tried an unknown tool." {
		t.Errorf("result = %q", result)
	}

	// Check that error message was sent for unknown tool
	msgs := mock.messages[1]
	var found bool
	for _, m := range msgs {
		if m.Role == llm.RoleTool && m.ToolCallID == "call-1" {
			if m.Content == "" {
				t.Error("expected error content for unknown tool")
			}
			found = true
		}
	}
	if !found {
		t.Error("expected tool error message in second call")
	}
}

func TestAgent_HistoryTrimming(t *testing.T) {
	store := newMockStore()

	ag := New(Config{
		LLMClient: &mockLLMClient{
			responses: make([]*llm.Response, 100),
		},
		STM:        memai.NewSTM(memai.STMConfig{}),
		LTM:        memai.NewLTM[string](store, dummyEmbedder, memai.LTMConfig{}),
		Store:      store,
		Tools:      tools.NewRegistry(),
		Embedder:   dummyEmbedder,
		ContextDir: "/tmp",
	})

	// Fill responses
	for i := range ag.llmClient.(*mockLLMClient).responses {
		ag.llmClient.(*mockLLMClient).responses[i] = &llm.Response{
			Message: llm.Message{Role: llm.RoleAssistant, Content: fmt.Sprintf("resp-%d", i)},
		}
	}

	// Run 30 times to exceed the 40-message history limit
	for i := 0; i < 30; i++ {
		_, err := ag.Run(context.Background(), fmt.Sprintf("msg-%d", i))
		if err != nil {
			t.Fatalf("Run %d: %v", i, err)
		}
	}

	if len(ag.history) > 40 {
		t.Errorf("history length = %d, want <= 40", len(ag.history))
	}
}

func TestAgent_PlanFlow(t *testing.T) {
	planHolder := &tools.PlanHolder{}

	mock := &mockLLMClient{
		responses: []*llm.Response{
			// LLM creates a plan
			{Message: llm.Message{
				Role: llm.RoleAssistant,
				ToolCalls: []llm.ToolCall{
					{ID: "call-1", Name: "create_plan", Arguments: `{"title":"Refactor","steps":["analyze","refactor","test"]}`},
				},
			}},
		},
	}
	store := newMockStore()

	ag := New(Config{
		LLMClient:  mock,
		STM:        memai.NewSTM(memai.STMConfig{}),
		LTM:        memai.NewLTM[string](store, dummyEmbedder, memai.LTMConfig{}),
		Store:      store,
		Tools:      tools.NewRegistry(tools.NewCreatePlan(planHolder), tools.NewUpdatePlanStep(planHolder)),
		PlanHolder: planHolder,
		Embedder:   dummyEmbedder,
		ContextDir: "/tmp",
	})

	result, err := ag.Run(context.Background(), "refactor the codebase")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Plan should be created and pending
	if !ag.HasPendingPlan() {
		t.Error("expected pending plan")
	}
	if !contains(result, "Refactor") {
		t.Errorf("result should contain plan title, got: %s", result)
	}

	// Approve and verify
	ag.ApprovePlan()
	if ag.HasPendingPlan() {
		t.Error("plan should no longer be pending after approval")
	}
}

func TestAgent_PlanReject(t *testing.T) {
	planHolder := &tools.PlanHolder{}

	mock := &mockLLMClient{
		responses: []*llm.Response{
			{Message: llm.Message{
				Role: llm.RoleAssistant,
				ToolCalls: []llm.ToolCall{
					{ID: "call-1", Name: "create_plan", Arguments: `{"title":"Bad Plan","steps":["step1"]}`},
				},
			}},
		},
	}
	store := newMockStore()

	ag := New(Config{
		LLMClient:  mock,
		STM:        memai.NewSTM(memai.STMConfig{}),
		LTM:        memai.NewLTM[string](store, dummyEmbedder, memai.LTMConfig{}),
		Store:      store,
		Tools:      tools.NewRegistry(tools.NewCreatePlan(planHolder)),
		PlanHolder: planHolder,
		Embedder:   dummyEmbedder,
		ContextDir: "/tmp",
	})

	ag.Run(context.Background(), "do something big")

	ag.RejectPlan()
	if ag.HasPendingPlan() {
		t.Error("plan should be cleared after rejection")
	}
	if ag.CurrentPlan() != "" {
		t.Error("CurrentPlan should be empty after rejection")
	}
}

func TestAgent_ContextCompression(t *testing.T) {
	store := newMockStore()

	// Create a mock that returns tool calls for many iterations, then a final response
	responses := make([]*llm.Response, 0)
	for i := 0; i < 15; i++ {
		responses = append(responses, &llm.Response{
			Message: llm.Message{
				Role: llm.RoleAssistant,
				ToolCalls: []llm.ToolCall{
					{ID: fmt.Sprintf("call-%d", i), Name: "read_file", Arguments: fmt.Sprintf(`{"path":"file%d.txt"}`, i)},
				},
			},
		})
	}
	responses = append(responses, &llm.Response{
		Message: llm.Message{Role: llm.RoleAssistant, Content: "Done reading all files"},
	})

	mock := &mockLLMClient{responses: responses}

	// Create a tool that returns large results
	bigTool := &fakeTool{name: "read_file", result: strings.Repeat("x", 5000)}

	ag := New(Config{
		LLMClient:         mock,
		STM:               memai.NewSTM(memai.STMConfig{}),
		LTM:               memai.NewLTM[string](store, dummyEmbedder, memai.LTMConfig{}),
		Store:             store,
		Tools:             tools.NewRegistry(bigTool),
		Embedder:          dummyEmbedder,
		ContextDir:        "/tmp",
		MaxToolIterations: 50,
		MaxContextChars:   20000, // Low threshold to trigger compression
	})

	result, err := ag.Run(context.Background(), "read all files")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result != "Done reading all files" {
		t.Errorf("result = %q", result)
	}

	// Verify compression happened: later LLM calls should have conversation summary
	if mock.callCount < 10 {
		t.Errorf("expected many LLM calls, got %d", mock.callCount)
	}

	// Check that a conversation summary message was inserted
	lastMsgs := mock.messages[mock.callCount-1]
	hasSummary := false
	for _, m := range lastMsgs {
		if m.Role == llm.RoleSystem && strings.Contains(m.Content, "Conversation Summary") {
			hasSummary = true
			break
		}
	}
	if !hasSummary {
		t.Error("expected a Conversation Summary message after compression")
	}
}

func TestAgent_WrapUpHint(t *testing.T) {
	store := newMockStore()

	maxIter := 5
	responses := make([]*llm.Response, 0)
	for i := 0; i < maxIter; i++ {
		responses = append(responses, &llm.Response{
			Message: llm.Message{
				Role: llm.RoleAssistant,
				ToolCalls: []llm.ToolCall{
					{ID: fmt.Sprintf("call-%d", i), Name: "read_file", Arguments: `{"path":"test.txt"}`},
				},
			},
		})
	}
	// Final response after exhausting iterations shouldn't happen since we exhaust the loop
	// But add one just in case
	responses = append(responses, &llm.Response{
		Message: llm.Message{Role: llm.RoleAssistant, Content: "wrapping up"},
	})

	mock := &mockLLMClient{responses: responses}
	smallTool := &fakeTool{name: "read_file", result: "content"}

	ag := New(Config{
		LLMClient:         mock,
		STM:               memai.NewSTM(memai.STMConfig{}),
		LTM:               memai.NewLTM[string](store, dummyEmbedder, memai.LTMConfig{}),
		Store:             store,
		Tools:             tools.NewRegistry(smallTool),
		Embedder:          dummyEmbedder,
		ContextDir:        "/tmp",
		MaxToolIterations: maxIter,
	})

	ag.Run(context.Background(), "do work")

	// Check that a system hint was injected near the end
	foundHint := false
	for _, callMsgs := range mock.messages {
		for _, m := range callMsgs {
			if m.Role == llm.RoleSystem && strings.Contains(m.Content, "remaining") {
				foundHint = true
			}
		}
	}
	if !foundHint {
		t.Error("expected wrap-up hint to be injected before iteration limit")
	}
}

// fakeTool implements tools.Tool for testing
type fakeTool struct {
	name   string
	result string
}

func (f *fakeTool) Name() string                                 { return f.name }
func (f *fakeTool) Description() string                          { return "fake tool" }
func (f *fakeTool) Parameters() map[string]interface{}            { return map[string]interface{}{"type": "object", "properties": map[string]interface{}{}} }
func (f *fakeTool) Execute(_ interface{}, _ string) (string, error) { return f.result, nil }

func TestCompressToolCallArgs(t *testing.T) {
	t.Run("write_file with large content", func(t *testing.T) {
		content := strings.Repeat("line\n", 100) // 500 chars, 100 lines
		tc := &llm.ToolCall{
			ID:        "call-1",
			Name:      "write_file",
			Arguments: fmt.Sprintf(`{"path":"main.go","content":%q}`, content),
		}
		compressToolCallArgs(tc, "File written successfully")

		if strings.Contains(tc.Arguments, "line\n") {
			t.Error("expected content to be compressed, but original content still present")
		}
		if !strings.Contains(tc.Arguments, "main.go") {
			t.Error("expected path to be preserved")
		}
		if !strings.Contains(tc.Arguments, "[written:") {
			t.Errorf("expected compressed marker, got: %s", tc.Arguments)
		}
	})

	t.Run("write_file with small content", func(t *testing.T) {
		tc := &llm.ToolCall{
			ID:        "call-1",
			Name:      "write_file",
			Arguments: `{"path":"small.txt","content":"hi"}`,
		}
		original := tc.Arguments
		compressToolCallArgs(tc, "File written successfully")

		if tc.Arguments != original {
			t.Error("small content should not be compressed")
		}
	})

	t.Run("non-write_file tool unchanged", func(t *testing.T) {
		tc := &llm.ToolCall{
			ID:        "call-1",
			Name:      "read_file",
			Arguments: `{"path":"main.go"}`,
		}
		original := tc.Arguments
		compressToolCallArgs(tc, "file content here")

		if tc.Arguments != original {
			t.Error("read_file args should not be modified")
		}
	})
}

func TestDeduplicateFileRead(t *testing.T) {
	store := newMockStore()
	ag := New(Config{
		LLMClient:  &mockLLMClient{},
		STM:        memai.NewSTM(memai.STMConfig{}),
		LTM:        memai.NewLTM[string](store, dummyEmbedder, memai.LTMConfig{}),
		Store:      store,
		Tools:      tools.NewRegistry(),
		Embedder:   dummyEmbedder,
		ContextDir: "/tmp",
	})

	args := `{"path":"/tmp/test.go"}`
	content := "package main\n\nfunc main() {}\n"

	// First read: should return full content
	result1 := ag.deduplicateFileRead(args, content, 5)
	if result1 != content {
		t.Errorf("first read should return full content, got: %s", result1)
	}

	// Second read with same content: should return placeholder
	result2 := ag.deduplicateFileRead(args, content, 10)
	if !strings.Contains(result2, "previously read") {
		t.Errorf("second read of same content should return placeholder, got: %s", result2)
	}
	if !strings.Contains(result2, "/tmp/test.go") {
		t.Error("placeholder should contain the file path")
	}

	// Third read with different content: should return full content
	newContent := "package main\n\nfunc main() { fmt.Println(\"hello\") }\n"
	result3 := ag.deduplicateFileRead(args, newContent, 15)
	if result3 != newContent {
		t.Errorf("read of changed content should return full content, got: %s", result3)
	}

	// Fourth read with same new content: should return placeholder again
	result4 := ag.deduplicateFileRead(args, newContent, 20)
	if !strings.Contains(result4, "previously read") {
		t.Errorf("repeated read of updated content should return placeholder, got: %s", result4)
	}
}

func TestCompressToolResult(t *testing.T) {
	t.Run("execute_command small output unchanged", func(t *testing.T) {
		result := "ok"
		got := compressToolResult("execute_command", result)
		if got != result {
			t.Errorf("small output should be unchanged, got: %s", got)
		}
	})

	t.Run("execute_command large output truncated", func(t *testing.T) {
		var lines []string
		for i := 0; i < 100; i++ {
			lines = append(lines, fmt.Sprintf("line %d: some test output here", i))
		}
		result := strings.Join(lines, "\n")
		got := compressToolResult("execute_command", result)
		if len(got) >= len(result) {
			t.Error("large command output should be compressed")
		}
		if !strings.Contains(got, "lines omitted") {
			t.Error("expected omission marker in compressed output")
		}
		if !strings.Contains(got, "line 0") {
			t.Error("expected first lines to be preserved")
		}
		if !strings.Contains(got, "line 99") {
			t.Error("expected last lines to be preserved")
		}
	})

	t.Run("web_fetch large output truncated", func(t *testing.T) {
		var lines []string
		for i := 0; i < 100; i++ {
			lines = append(lines, fmt.Sprintf("paragraph %d: %s", i, strings.Repeat("text ", 20)))
		}
		result := strings.Join(lines, "\n")
		got := compressToolResult("web_fetch", result)
		if len(got) >= len(result) {
			t.Error("large web_fetch output should be compressed")
		}
		if !strings.Contains(got, "lines omitted") {
			t.Error("expected omission marker")
		}
	})

	t.Run("read_file unchanged", func(t *testing.T) {
		result := strings.Repeat("code line\n", 100)
		got := compressToolResult("read_file", result)
		if got != result {
			t.Error("read_file should not be compressed by compressToolResult")
		}
	})
}

func TestStructuredContextWindow(t *testing.T) {
	store := newMockStore()

	t.Run("collapses old messages into summary", func(t *testing.T) {
		// Build 25 messages: system + 24 others
		messages := []llm.Message{
			{Role: llm.RoleSystem, Content: "system prompt"},
		}
		for i := 0; i < 12; i++ {
			messages = append(messages, llm.Message{
				Role:    llm.RoleUser,
				Content: fmt.Sprintf("Question %d about the code", i),
			})
			messages = append(messages, llm.Message{
				Role:    llm.RoleAssistant,
				Content: fmt.Sprintf("Answer %d with details about the implementation. %s", i, strings.Repeat("padding ", 30)),
			})
		}

		ag := New(Config{
			LLMClient:       &mockLLMClient{},
			STM:             memai.NewSTM(memai.STMConfig{}),
			LTM:             memai.NewLTM[string](store, dummyEmbedder, memai.LTMConfig{}),
			Store:           store,
			Tools:           tools.NewRegistry(),
			Embedder:        dummyEmbedder,
			ContextDir:      "/tmp",
			MaxContextChars: 500, // very low to force compression
		})

		compressed := ag.compressMessages(messages)

		// Result should be: system + summary + recent window
		if len(compressed) > recentWindowSize+2 {
			t.Errorf("expected at most %d messages (system+summary+recent), got %d", recentWindowSize+2, len(compressed))
		}

		// First message should be system prompt
		if compressed[0].Role != llm.RoleSystem || compressed[0].Content != "system prompt" {
			t.Error("first message should be original system prompt")
		}

		// Second message should be conversation summary
		if compressed[1].Role != llm.RoleSystem || !strings.Contains(compressed[1].Content, "Conversation Summary") {
			t.Errorf("second message should be conversation summary, got: %s", compressed[1].Content[:min(100, len(compressed[1].Content))])
		}

		// Agent should have accumulated summary
		if ag.conversationSummary == "" {
			t.Error("agent conversationSummary should be populated")
		}

		// Recent messages should be intact (last few from original)
		lastOriginal := messages[len(messages)-1].Content
		lastCompressed := compressed[len(compressed)-1].Content
		if lastOriginal != lastCompressed {
			t.Error("last message should be unchanged")
		}
	})

	t.Run("accumulates summary across compressions", func(t *testing.T) {
		ag := New(Config{
			LLMClient:       &mockLLMClient{},
			STM:             memai.NewSTM(memai.STMConfig{}),
			LTM:             memai.NewLTM[string](store, dummyEmbedder, memai.LTMConfig{}),
			Store:           store,
			Tools:           tools.NewRegistry(),
			Embedder:        dummyEmbedder,
			ContextDir:      "/tmp",
			MaxContextChars: 500,
		})

		// First compression
		msgs1 := []llm.Message{{Role: llm.RoleSystem, Content: "sys"}}
		for i := 0; i < 15; i++ {
			msgs1 = append(msgs1, llm.Message{Role: llm.RoleUser, Content: fmt.Sprintf("First batch msg %d. %s", i, strings.Repeat("pad ", 20))})
		}
		ag.compressMessages(msgs1)
		summary1 := ag.conversationSummary

		// Second compression with new messages
		msgs2 := []llm.Message{{Role: llm.RoleSystem, Content: "sys"}}
		for i := 0; i < 15; i++ {
			msgs2 = append(msgs2, llm.Message{Role: llm.RoleUser, Content: fmt.Sprintf("Second batch msg %d. %s", i, strings.Repeat("pad ", 20))})
		}
		ag.compressMessages(msgs2)
		summary2 := ag.conversationSummary

		// Second summary should be longer (accumulated)
		if len(summary2) <= len(summary1) {
			t.Errorf("accumulated summary should grow: first=%d, second=%d", len(summary1), len(summary2))
		}
	})

	t.Run("no compression below threshold", func(t *testing.T) {
		ag := New(Config{
			LLMClient:       &mockLLMClient{},
			STM:             memai.NewSTM(memai.STMConfig{}),
			LTM:             memai.NewLTM[string](store, dummyEmbedder, memai.LTMConfig{}),
			Store:           store,
			Tools:           tools.NewRegistry(),
			Embedder:        dummyEmbedder,
			ContextDir:      "/tmp",
			MaxContextChars: 100000,
		})

		messages := []llm.Message{
			{Role: llm.RoleSystem, Content: "sys"},
			{Role: llm.RoleUser, Content: "hello"},
			{Role: llm.RoleAssistant, Content: "hi"},
		}
		result := ag.compressMessages(messages)
		if len(result) != len(messages) {
			t.Error("messages below threshold should not be compressed")
		}
	})
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func TestFilteredToolDefs(t *testing.T) {
	planHolder := &tools.PlanHolder{}
	store := newMockStore()

	ag := New(Config{
		LLMClient:  &mockLLMClient{},
		STM:        memai.NewSTM(memai.STMConfig{}),
		LTM:        memai.NewLTM[string](store, dummyEmbedder, memai.LTMConfig{}),
		Store:      store,
		Tools:      tools.NewRegistry(tools.NewFileRead("/tmp"), tools.NewCreatePlan(planHolder), tools.NewUpdatePlanStep(planHolder), tools.NewShowPlan(planHolder)),
		PlanHolder: planHolder,
		Embedder:   dummyEmbedder,
		ContextDir: "/tmp",
	})

	t.Run("no plan excludes show_plan and update_plan_step", func(t *testing.T) {
		defs := ag.filteredToolDefs()
		names := make(map[string]bool)
		for _, d := range defs {
			names[d.Name] = true
		}
		if names["show_plan"] {
			t.Error("show_plan should be excluded when no plan exists")
		}
		if names["update_plan_step"] {
			t.Error("update_plan_step should be excluded when no plan exists")
		}
		if !names["create_plan"] {
			t.Error("create_plan should be included when no plan exists")
		}
	})

	t.Run("with plan excludes create_plan", func(t *testing.T) {
		planHolder.Current = &plan.Plan{}
		defs := ag.filteredToolDefs()
		names := make(map[string]bool)
		for _, d := range defs {
			names[d.Name] = true
		}
		if names["create_plan"] {
			t.Error("create_plan should be excluded when plan exists")
		}
		if !names["show_plan"] {
			t.Error("show_plan should be included when plan exists")
		}
		if !names["update_plan_step"] {
			t.Error("update_plan_step should be included when plan exists")
		}
		planHolder.Current = nil
	})
}

func TestMessagesCharCount(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleSystem, Content: "hello"},
		{Role: llm.RoleAssistant, Content: "world", ToolCalls: []llm.ToolCall{{Arguments: "abc"}}},
	}
	got := messagesCharCount(msgs)
	want := 5 + 5 + 3 // "hello" + "world" + "abc"
	if got != want {
		t.Errorf("messagesCharCount = %d, want %d", got, want)
	}
}

func TestBuildSystemPrompt(t *testing.T) {
	t.Run("with context and memory", func(t *testing.T) {
		prompt := buildSystemPrompt("testbot", "/home/user/project", "## Relevant Memories\n- memory1", "", "")
		if prompt == "" {
			t.Fatal("empty prompt")
		}
		if !contains(prompt, "testbot") {
			t.Error("expected 'testbot' in prompt")
		}
		if !contains(prompt, "/home/user/project") {
			t.Error("expected context dir in prompt")
		}
		if !contains(prompt, "memory1") {
			t.Error("expected memory context in prompt")
		}
	})

	t.Run("without memory", func(t *testing.T) {
		prompt := buildSystemPrompt("pmaid", "/tmp", "", "", "")
		if contains(prompt, "Relevant Memories") {
			t.Error("should not contain memories section when empty")
		}
	})

	t.Run("with plan context", func(t *testing.T) {
		prompt := buildSystemPrompt("pmaid", "/tmp", "", "## Active Plan\n1. step one", "")
		if !contains(prompt, "Active Plan") {
			t.Error("expected plan context in prompt")
		}
	})

	t.Run("contains planning instructions", func(t *testing.T) {
		prompt := buildSystemPrompt("pmaid", "/tmp", "", "", "")
		if !contains(prompt, "create_plan") {
			t.Error("expected planning instructions in prompt")
		}
	})
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
