package openai

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/ieee0824/pmaid/internal/llm"
	oai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
)

const (
	maxRetries    = 5
	baseDelay     = 1 * time.Second
	maxDelay      = 30 * time.Second
)

type Client struct {
	client oai.Client
	model  oai.ChatModel
}

func New(model string, apiKey string, opts ...Option) *Client {
	var reqOpts []option.RequestOption
	if apiKey != "" {
		reqOpts = append(reqOpts, option.WithAPIKey(apiKey))
	}
	for _, o := range opts {
		if o.BaseURL != "" {
			reqOpts = append(reqOpts, option.WithBaseURL(o.BaseURL))
		}
	}
	c := oai.NewClient(reqOpts...)
	return &Client{client: c, model: oai.ChatModel(model)}
}

// Option holds optional client configuration.
type Option struct {
	BaseURL string
}

// WithBaseURL returns an Option that sets a custom API base URL.
func WithBaseURL(url string) Option {
	return Option{BaseURL: url}
}

func (c *Client) Chat(ctx context.Context, messages []llm.Message, tools []llm.ToolDef) (*llm.Response, error) {
	params := oai.ChatCompletionNewParams{
		Messages: convertMessages(messages),
		Model:    c.model,
	}

	if len(tools) > 0 {
		params.Tools = convertTools(tools)
	}

	var completion *oai.ChatCompletion
	var lastErr error

	for attempt := range maxRetries {
		completion, lastErr = c.client.Chat.Completions.New(ctx, params)
		if lastErr == nil {
			break
		}
		if !isRetryable(lastErr) {
			return nil, fmt.Errorf("openai chat: %w", lastErr)
		}
		delay := time.Duration(math.Min(
			float64(baseDelay)*math.Pow(2, float64(attempt)),
			float64(maxDelay),
		))
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(delay):
		}
	}
	if lastErr != nil {
		return nil, fmt.Errorf("openai chat (after %d retries): %w", maxRetries, lastErr)
	}

	if len(completion.Choices) == 0 {
		return nil, fmt.Errorf("openai: no choices returned")
	}

	choice := completion.Choices[0]
	msg := llm.Message{
		Role:    llm.RoleAssistant,
		Content: choice.Message.Content,
	}

	for _, tc := range choice.Message.ToolCalls {
		msg.ToolCalls = append(msg.ToolCalls, llm.ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		})
	}

	return &llm.Response{
		Message: msg,
		Usage: llm.Usage{
			PromptTokens:     int(completion.Usage.PromptTokens),
			CompletionTokens: int(completion.Usage.CompletionTokens),
		},
	}, nil
}

func isRetryable(err error) bool {
	var apiErr *oai.Error
	if errors.As(err, &apiErr) {
		// 429 but insufficient_quota is not retryable
		if apiErr.StatusCode == http.StatusTooManyRequests && isQuotaExceeded(apiErr) {
			return false
		}
		switch apiErr.StatusCode {
		case http.StatusTooManyRequests, http.StatusInternalServerError,
			http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
			return true
		}
	}
	return false
}

func isQuotaExceeded(apiErr *oai.Error) bool {
	msg := strings.ToLower(apiErr.Error())
	return strings.Contains(msg, "insufficient_quota") || strings.Contains(msg, "exceeded your current quota")
}

func convertMessages(messages []llm.Message) []oai.ChatCompletionMessageParamUnion {
	out := make([]oai.ChatCompletionMessageParamUnion, 0, len(messages))
	for _, m := range messages {
		switch m.Role {
		case llm.RoleSystem:
			out = append(out, oai.SystemMessage(m.Content))
		case llm.RoleUser:
			out = append(out, oai.UserMessage(m.Content))
		case llm.RoleAssistant:
			if len(m.ToolCalls) > 0 {
				assistantMsg := oai.ChatCompletionAssistantMessageParam{
					Content: oai.ChatCompletionAssistantMessageParamContentUnion{
						OfString: param.NewOpt(m.Content),
					},
				}
				for _, tc := range m.ToolCalls {
					assistantMsg.ToolCalls = append(assistantMsg.ToolCalls, oai.ChatCompletionMessageToolCallUnionParam{
						OfFunction: &oai.ChatCompletionMessageFunctionToolCallParam{
							ID: tc.ID,
							Function: oai.ChatCompletionMessageFunctionToolCallFunctionParam{
								Name:      tc.Name,
								Arguments: tc.Arguments,
							},
						},
					})
				}
				out = append(out, oai.ChatCompletionMessageParamUnion{
					OfAssistant: &assistantMsg,
				})
			} else {
				out = append(out, oai.AssistantMessage(m.Content))
			}
		case llm.RoleTool:
			out = append(out, oai.ToolMessage(m.Content, m.ToolCallID))
		}
	}
	return out
}

func convertTools(tools []llm.ToolDef) []oai.ChatCompletionToolUnionParam {
	out := make([]oai.ChatCompletionToolUnionParam, 0, len(tools))
	for _, t := range tools {
		out = append(out, oai.ChatCompletionFunctionTool(oai.FunctionDefinitionParam{
			Name:        t.Name,
			Description: oai.String(t.Description),
			Parameters:  oai.FunctionParameters(t.Parameters),
		}))
	}
	return out
}
