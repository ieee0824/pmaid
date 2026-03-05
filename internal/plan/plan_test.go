package plan

import (
	"strings"
	"testing"
)

func TestNew(t *testing.T) {
	p := New("Refactoring", []string{"analyze code", "extract functions", "add tests"})

	if p.Title != "Refactoring" {
		t.Errorf("Title = %q", p.Title)
	}
	if len(p.Steps) != 3 {
		t.Fatalf("len(Steps) = %d, want 3", len(p.Steps))
	}
	for i, s := range p.Steps {
		if s.ID != i+1 {
			t.Errorf("Step[%d].ID = %d", i, s.ID)
		}
		if s.Status != StatusPending {
			t.Errorf("Step[%d].Status = %q, want pending", i, s.Status)
		}
	}
	if p.Approved {
		t.Error("new plan should not be approved")
	}
}

func TestApprove(t *testing.T) {
	p := New("Test", []string{"step 1"})

	if p.IsApproved() {
		t.Error("should not be approved initially")
	}
	p.Approve()
	if !p.IsApproved() {
		t.Error("should be approved after Approve()")
	}
}

func TestSetStepStatus(t *testing.T) {
	p := New("Test", []string{"step 1", "step 2"})

	if err := p.SetStepStatus(1, StatusInProgress); err != nil {
		t.Fatalf("SetStepStatus: %v", err)
	}
	if p.Steps[0].Status != StatusInProgress {
		t.Errorf("Status = %q, want in_progress", p.Steps[0].Status)
	}

	if err := p.SetStepStatus(1, StatusCompleted); err != nil {
		t.Fatalf("SetStepStatus: %v", err)
	}
	if p.Steps[0].Status != StatusCompleted {
		t.Errorf("Status = %q, want completed", p.Steps[0].Status)
	}

	if err := p.SetStepStatus(99, StatusCompleted); err == nil {
		t.Error("expected error for invalid step ID")
	}
}

func TestCurrentStep(t *testing.T) {
	p := New("Test", []string{"step 1", "step 2", "step 3"})

	cur := p.CurrentStep()
	if cur == nil || cur.ID != 1 {
		t.Errorf("CurrentStep = %v, want step 1", cur)
	}

	p.SetStepStatus(1, StatusCompleted)
	cur = p.CurrentStep()
	if cur == nil || cur.ID != 2 {
		t.Errorf("CurrentStep = %v, want step 2", cur)
	}

	p.SetStepStatus(2, StatusCompleted)
	p.SetStepStatus(3, StatusCompleted)
	cur = p.CurrentStep()
	if cur != nil {
		t.Errorf("CurrentStep = %v, want nil", cur)
	}
}

func TestIsComplete(t *testing.T) {
	p := New("Test", []string{"step 1", "step 2"})

	if p.IsComplete() {
		t.Error("should not be complete initially")
	}

	p.SetStepStatus(1, StatusCompleted)
	if p.IsComplete() {
		t.Error("should not be complete with pending steps")
	}

	p.SetStepStatus(2, StatusSkipped)
	if !p.IsComplete() {
		t.Error("should be complete when all done/skipped")
	}
}

func TestProgress(t *testing.T) {
	p := New("Test", []string{"a", "b", "c"})

	completed, total := p.Progress()
	if completed != 0 || total != 3 {
		t.Errorf("Progress = %d/%d, want 0/3", completed, total)
	}

	p.SetStepStatus(1, StatusCompleted)
	p.SetStepStatus(2, StatusSkipped)
	completed, total = p.Progress()
	if completed != 2 || total != 3 {
		t.Errorf("Progress = %d/%d, want 2/3", completed, total)
	}
}

func TestString(t *testing.T) {
	p := New("My Plan", []string{"first", "second"})
	p.SetStepStatus(1, StatusCompleted)

	s := p.String()
	if !strings.Contains(s, "My Plan") {
		t.Error("should contain title")
	}
	if !strings.Contains(s, "✓") {
		t.Error("should contain checkmark for completed")
	}
	if !strings.Contains(s, "1/2") {
		t.Error("should show progress 1/2")
	}
}
