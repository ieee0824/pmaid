package tools

import (
	"github.com/ieee0824/pmaid/internal/llm"
)

type Tool interface {
	Name() string
	Description() string
	Parameters() map[string]interface{}
	Execute(ctx interface{}, args string) (string, error)
}

type Registry struct {
	tools map[string]Tool
}

func NewRegistry(tt ...Tool) *Registry {
	r := &Registry{tools: make(map[string]Tool)}
	for _, t := range tt {
		r.tools[t.Name()] = t
	}
	return r
}

func (r *Registry) Get(name string) (Tool, bool) {
	t, ok := r.tools[name]
	return t, ok
}

func (r *Registry) Definitions() []llm.ToolDef {
	defs := make([]llm.ToolDef, 0, len(r.tools))
	for _, t := range r.tools {
		defs = append(defs, llm.ToolDef{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  t.Parameters(),
		})
	}
	return defs
}

func (r *Registry) List() []Tool {
	tt := make([]Tool, 0, len(r.tools))
	for _, t := range r.tools {
		tt = append(tt, t)
	}
	return tt
}
