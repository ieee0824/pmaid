package memory

import (
	"context"
	"math"
	"testing"
)

func TestTokenize(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"hello world", []string{"hello", "world"}},
		{"Hello, World!", []string{"hello", "world"}},
		{"foo123 bar", []string{"foo123", "bar"}},
		{"", nil},
		{"  spaces  ", []string{"spaces"}},
		{"日本語テスト", []string{"日本語テスト"}},
	}

	for _, tt := range tests {
		got := tokenize(tt.input)
		if len(got) != len(tt.want) {
			t.Errorf("tokenize(%q) = %v, want %v", tt.input, got, tt.want)
			continue
		}
		for i := range got {
			if got[i] != tt.want[i] {
				t.Errorf("tokenize(%q)[%d] = %q, want %q", tt.input, i, got[i], tt.want[i])
			}
		}
	}
}

func TestNewTFIDFEmbedder(t *testing.T) {
	e := NewTFIDFEmbedder(64)
	if e.dim != 64 {
		t.Errorf("dim = %d, want 64", e.dim)
	}
	if e.docCount != 0 {
		t.Errorf("docCount = %d, want 0", e.docCount)
	}
}

func TestEmbed_FixedDimension(t *testing.T) {
	e := NewTFIDFEmbedder(128)
	ctx := context.Background()

	vec, err := e.Embed(ctx, "hello world foo bar")
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}
	if len(vec) != 128 {
		t.Errorf("len(vec) = %d, want 128", len(vec))
	}
}

func TestEmbed_EmptyInput(t *testing.T) {
	e := NewTFIDFEmbedder(64)
	ctx := context.Background()

	vec, err := e.Embed(ctx, "")
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}
	if len(vec) != 64 {
		t.Errorf("len(vec) = %d, want 64", len(vec))
	}
	for i, v := range vec {
		if v != 0 {
			t.Errorf("vec[%d] = %f, want 0 for empty input", i, v)
		}
	}
}

func TestEmbed_Normalized(t *testing.T) {
	e := NewTFIDFEmbedder(256)
	ctx := context.Background()

	vec, err := e.Embed(ctx, "the quick brown fox jumps over the lazy dog")
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}

	var norm float64
	for _, v := range vec {
		norm += v * v
	}
	norm = math.Sqrt(norm)

	if math.Abs(norm-1.0) > 1e-9 {
		t.Errorf("L2 norm = %f, want 1.0", norm)
	}
}

func TestEmbed_SimilarTextsSimilarVectors(t *testing.T) {
	e := NewTFIDFEmbedder(512)
	ctx := context.Background()

	v1, _ := e.Embed(ctx, "go programming language")
	v2, _ := e.Embed(ctx, "go programming tutorial")
	v3, _ := e.Embed(ctx, "banana apple fruit salad")

	sim12 := cosine(v1, v2)
	sim13 := cosine(v1, v3)

	if sim12 <= sim13 {
		t.Errorf("similar texts should have higher similarity: sim(1,2)=%f <= sim(1,3)=%f", sim12, sim13)
	}
}

func TestEmbed_DocFreqUpdated(t *testing.T) {
	e := NewTFIDFEmbedder(64)
	ctx := context.Background()

	e.Embed(ctx, "hello world")
	e.Embed(ctx, "hello go")

	if e.docCount != 2 {
		t.Errorf("docCount = %d, want 2", e.docCount)
	}
	if e.docFreq["hello"] != 2 {
		t.Errorf("docFreq[hello] = %d, want 2", e.docFreq["hello"])
	}
	if e.docFreq["world"] != 1 {
		t.Errorf("docFreq[world] = %d, want 1", e.docFreq["world"])
	}
}

func TestEmbedFunc(t *testing.T) {
	e := NewTFIDFEmbedder(64)
	fn := e.EmbedFunc()

	vec, err := fn(context.Background(), "test")
	if err != nil {
		t.Fatalf("EmbedFunc error: %v", err)
	}
	if len(vec) != 64 {
		t.Errorf("len(vec) = %d, want 64", len(vec))
	}
}

func cosine(a, b []float64) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	d := math.Sqrt(na) * math.Sqrt(nb)
	if d == 0 {
		return 0
	}
	return dot / d
}
