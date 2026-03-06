package ui

import (
	"os"
	"strings"

	"github.com/fatih/color"
	"github.com/mattn/go-isatty"
)

// ColorEnabled returns whether colored output should be enabled.
// Rules:
// - PMAID_FORCE_COLOR=1 => force enable
// - NO_COLOR set => disable
// - TERM=dumb => disable
// - non-TTY stdout => disable
func ColorEnabled() bool {
	if v := strings.TrimSpace(os.Getenv("PMAID_FORCE_COLOR")); v != "" && v != "0" {
		return true
	}
	if _, ok := os.LookupEnv("NO_COLOR"); ok {
		return false
	}
	if strings.EqualFold(os.Getenv("TERM"), "dumb") {
		return false
	}
	// If stdout isn't a TTY, avoid emitting ANSI escapes.
	if !isatty.IsTerminal(os.Stdout.Fd()) {
		return false
	}
	return true
}

// Styles holds color styles used in CLI output.
// When Enabled=false, all functions return plain strings.
type Styles struct {
	Enabled bool

	Banner   func(a ...any) string
	PromptMe func(a ...any) string
	PromptAI func(a ...any) string
	Warn     func(a ...any) string
	Info     func(a ...any) string
	Error    func(a ...any) string
}

func NewStyles() Styles {
	enabled := ColorEnabled()
	if !enabled {
		plain := color.New(color.Reset).SprintFunc()
		return Styles{
			Enabled:  false,
			Banner:   plain,
			PromptMe: plain,
			PromptAI: plain,
			Warn:     plain,
			Info:     plain,
			Error:    plain,
		}
	}

	return Styles{
		Enabled:  true,
		Banner:   color.New(color.FgHiCyan, color.Bold).SprintFunc(),
		PromptMe: color.New(color.FgHiGreen, color.Bold).SprintFunc(),
		PromptAI: color.New(color.FgHiMagenta, color.Bold).SprintFunc(),
		Warn:     color.New(color.FgHiYellow, color.Bold).SprintFunc(),
		Info:     color.New(color.FgHiBlue).SprintFunc(),
		Error:    color.New(color.FgHiRed, color.Bold).SprintFunc(),
	}
}
