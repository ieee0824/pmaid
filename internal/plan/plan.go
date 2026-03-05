package plan

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

type StepStatus string

const (
	StatusPending    StepStatus = "pending"
	StatusInProgress StepStatus = "in_progress"
	StatusCompleted  StepStatus = "completed"
	StatusSkipped    StepStatus = "skipped"
)

type Step struct {
	ID          int        `json:"id"`
	Description string     `json:"description"`
	Status      StepStatus `json:"status"`
}

type Plan struct {
	mu        sync.RWMutex
	Title     string    `json:"title"`
	Steps     []Step    `json:"steps"`
	CreatedAt time.Time `json:"created_at"`
	Approved  bool      `json:"approved"`
}

func New(title string, descriptions []string) *Plan {
	steps := make([]Step, len(descriptions))
	for i, d := range descriptions {
		steps[i] = Step{
			ID:          i + 1,
			Description: d,
			Status:      StatusPending,
		}
	}
	return &Plan{
		Title:     title,
		Steps:     steps,
		CreatedAt: time.Now(),
	}
}

func (p *Plan) Approve() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.Approved = true
}

func (p *Plan) IsApproved() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.Approved
}

func (p *Plan) SetStepStatus(id int, status StepStatus) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	for i := range p.Steps {
		if p.Steps[i].ID == id {
			p.Steps[i].Status = status
			return nil
		}
	}
	return fmt.Errorf("step %d not found", id)
}

func (p *Plan) CurrentStep() *Step {
	p.mu.RLock()
	defer p.mu.RUnlock()
	for i := range p.Steps {
		if p.Steps[i].Status == StatusPending || p.Steps[i].Status == StatusInProgress {
			return &p.Steps[i]
		}
	}
	return nil
}

func (p *Plan) IsComplete() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	for _, s := range p.Steps {
		if s.Status == StatusPending || s.Status == StatusInProgress {
			return false
		}
	}
	return true
}

func (p *Plan) Progress() (completed, total int) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	total = len(p.Steps)
	for _, s := range p.Steps {
		if s.Status == StatusCompleted || s.Status == StatusSkipped {
			completed++
		}
	}
	return
}

func (p *Plan) String() string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("# %s\n\n", p.Title))

	for _, s := range p.Steps {
		icon := "  "
		switch s.Status {
		case StatusCompleted:
			icon = "✓ "
		case StatusInProgress:
			icon = "→ "
		case StatusSkipped:
			icon = "- "
		case StatusPending:
			icon = "  "
		}
		sb.WriteString(fmt.Sprintf("%s%d. %s\n", icon, s.ID, s.Description))
	}

	completed, total := 0, len(p.Steps)
	for _, s := range p.Steps {
		if s.Status == StatusCompleted || s.Status == StatusSkipped {
			completed++
		}
	}
	sb.WriteString(fmt.Sprintf("\n進捗: %d/%d\n", completed, total))

	return sb.String()
}
