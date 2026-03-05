package agent

import (
	"context"
	"fmt"
	"testing"

	memai "github.com/ieee0824/memAI-go"
	"github.com/ieee0824/pmaid/internal/llm"
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

func TestBuildSystemPrompt(t *testing.T) {
	t.Run("with context and memory", func(t *testing.T) {
		prompt := buildSystemPrompt("/home/user/project", "## Relevant Memories\n- memory1")
		if prompt == "" {
			t.Fatal("empty prompt")
		}
		if !contains(prompt, "pmaid") {
			t.Error("expected 'pmaid' in prompt")
		}
		if !contains(prompt, "/home/user/project") {
			t.Error("expected context dir in prompt")
		}
		if !contains(prompt, "memory1") {
			t.Error("expected memory context in prompt")
		}
	})

	t.Run("without memory", func(t *testing.T) {
		prompt := buildSystemPrompt("/tmp", "")
		if contains(prompt, "Relevant Memories") {
			t.Error("should not contain memories section when empty")
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
