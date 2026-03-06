package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"golang.org/x/net/html"
)

type WebFetch struct {
	client *http.Client
}

func NewWebFetch() *WebFetch {
	return &WebFetch{
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (w *WebFetch) Name() string        { return "web_fetch" }
func (w *WebFetch) Description() string  { return "Fetch a web page and return its text content (HTML tags stripped)" }
func (w *WebFetch) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"url": map[string]interface{}{
				"type":        "string",
				"description": "URL to fetch",
			},
		},
		"required": []string{"url"},
	}
}

func (w *WebFetch) Execute(ctxRaw interface{}, args string) (string, error) {
	var params struct {
		URL string `json:"url"`
	}
	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("parse args: %w", err)
	}

	if params.URL == "" {
		return "", fmt.Errorf("url is required")
	}
	if !strings.HasPrefix(params.URL, "http://") && !strings.HasPrefix(params.URL, "https://") {
		return "", fmt.Errorf("url must start with http:// or https://")
	}

	ctx, ok := ctxRaw.(context.Context)
	if !ok {
		ctx = context.Background()
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, params.URL, nil)
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("User-Agent", "pmaid/1.0")

	resp, err := w.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d %s", resp.StatusCode, resp.Status)
	}

	const maxBody = 1024 * 1024 // 1MB
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxBody))
	if err != nil {
		return "", fmt.Errorf("read body: %w", err)
	}

	contentType := resp.Header.Get("Content-Type")
	if strings.Contains(contentType, "text/html") {
		text := extractText(string(body))
		const maxText = 100 * 1024
		if len(text) > maxText {
			return text[:maxText] + "\n... (truncated)", nil
		}
		return text, nil
	}

	// Non-HTML: return raw text
	content := string(body)
	const maxText = 100 * 1024
	if len(content) > maxText {
		return content[:maxText] + "\n... (truncated)", nil
	}
	return content, nil
}

// extractText parses HTML and returns visible text content.
func extractText(htmlStr string) string {
	doc, err := html.Parse(strings.NewReader(htmlStr))
	if err != nil {
		return htmlStr
	}

	var sb strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		// Skip script, style, and other non-visible elements
		if n.Type == html.ElementNode {
			switch n.Data {
			case "script", "style", "noscript", "head":
				return
			}
		}

		if n.Type == html.TextNode {
			text := strings.TrimSpace(n.Data)
			if text != "" {
				sb.WriteString(text)
				sb.WriteString(" ")
			}
		}

		// Add newline after block elements
		if n.Type == html.ElementNode {
			switch n.Data {
			case "p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6",
				"li", "tr", "blockquote", "pre", "hr", "section", "article":
				sb.WriteString("\n")
			}
		}

		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}

		if n.Type == html.ElementNode {
			switch n.Data {
			case "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
				"li", "tr", "blockquote", "pre", "section", "article":
				sb.WriteString("\n")
			}
		}
	}
	walk(doc)

	// Clean up excessive whitespace
	result := sb.String()
	for strings.Contains(result, "\n\n\n") {
		result = strings.ReplaceAll(result, "\n\n\n", "\n\n")
	}
	return strings.TrimSpace(result)
}
