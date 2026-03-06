package tools

import (
	"encoding/json"
	"fmt"

	"github.com/ieee0824/pmaid/internal/plan"
)

// PlanHolder holds a reference to the current plan.
// The agent sets this so that plan tools can access it.
type PlanHolder struct {
	Current *plan.Plan
}

// --- create_plan tool ---

type CreatePlan struct {
	holder *PlanHolder
}

func NewCreatePlan(holder *PlanHolder) *CreatePlan {
	return &CreatePlan{holder: holder}
}

func (c *CreatePlan) Name() string        { return "create_plan" }
func (c *CreatePlan) Description() string {
	return "Create an execution plan for a large task. The plan must be approved by the user before execution begins. Use this for tasks that involve multiple files, complex refactoring, or multi-step operations. The first step of the plan MUST be reading existing code to understand the codebase conventions (naming, error handling, code style, patterns) before making any changes."
}
func (c *CreatePlan) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"title": map[string]interface{}{
				"type":        "string",
				"description": "Short title describing the task",
			},
			"steps": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "string",
				},
				"description": "Ordered list of steps to execute",
			},
		},
		"required": []string{"title", "steps"},
	}
}

func (c *CreatePlan) Execute(_ interface{}, args string) (string, error) {
	var params struct {
		Title string   `json:"title"`
		Steps []string `json:"steps"`
	}
	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("parse args: %w", err)
	}
	if len(params.Steps) == 0 {
		return "", fmt.Errorf("plan must have at least one step")
	}

	p := plan.New(params.Title, params.Steps)
	c.holder.Current = p

	return fmt.Sprintf("Plan created. Waiting for user approval.\n\n%s", p.String()), nil
}

// --- update_plan_step tool ---

type UpdatePlanStep struct {
	holder *PlanHolder
}

func NewUpdatePlanStep(holder *PlanHolder) *UpdatePlanStep {
	return &UpdatePlanStep{holder: holder}
}

func (u *UpdatePlanStep) Name() string        { return "update_plan_step" }
func (u *UpdatePlanStep) Description() string {
	return "Update the status of a plan step. Use 'in_progress' when starting a step and 'completed' when done."
}
func (u *UpdatePlanStep) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"step_id": map[string]interface{}{
				"type":        "integer",
				"description": "The step number to update (1-based)",
			},
			"status": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"in_progress", "completed", "skipped"},
				"description": "New status for the step",
			},
		},
		"required": []string{"step_id", "status"},
	}
}

func (u *UpdatePlanStep) Execute(_ interface{}, args string) (string, error) {
	var params struct {
		StepID int    `json:"step_id"`
		Status string `json:"status"`
	}
	if err := json.Unmarshal([]byte(args), &params); err != nil {
		return "", fmt.Errorf("parse args: %w", err)
	}

	if u.holder.Current == nil {
		return "", fmt.Errorf("no active plan")
	}

	status := plan.StepStatus(params.Status)
	switch status {
	case plan.StatusInProgress, plan.StatusCompleted, plan.StatusSkipped:
	default:
		return "", fmt.Errorf("invalid status: %q", params.Status)
	}

	if err := u.holder.Current.SetStepStatus(params.StepID, status); err != nil {
		return "", err
	}

	return u.holder.Current.String(), nil
}

// --- show_plan tool ---

type ShowPlan struct {
	holder *PlanHolder
}

func NewShowPlan(holder *PlanHolder) *ShowPlan {
	return &ShowPlan{holder: holder}
}

func (s *ShowPlan) Name() string        { return "show_plan" }
func (s *ShowPlan) Description() string { return "Show the current execution plan and its progress" }
func (s *ShowPlan) Parameters() map[string]interface{} {
	return map[string]interface{}{
		"type":       "object",
		"properties": map[string]interface{}{},
	}
}

func (s *ShowPlan) Execute(_ interface{}, _ string) (string, error) {
	if s.holder.Current == nil {
		return "No active plan.", nil
	}
	return s.holder.Current.String(), nil
}
