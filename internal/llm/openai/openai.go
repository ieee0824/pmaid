package openai

import (
	"context"
	"fmt"

	"github.com/ieee0824/pmaid/internal/llm"
	oai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
)

type Client struct {
	client oai.Client
	model  oai.ChatModel
}

func New(model string, apiKey string) *Client {
	var opts []option.RequestOption
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}
	c := oai.NewClient(opts...)
	return &Client{client: c, model: oai.ChatModel(model)}
}

func (c *Client) Chat(ctx context.Context, messages []llm.Message, tools []llm.ToolDef) (*llm.Response, error) {
	params := oai.ChatCompletionNewParams{
		Messages: convertMessages(messages),
		Model:    c.model,
	}

	if len(tools) > 0 {
		params.Tools = convertTools(tools)
	}

	completion, err := c.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("openai chat: %w", err)
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
