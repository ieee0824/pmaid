package memory

import (
	"context"
	"hash/fnv"
	"math"
	"strings"
	"sync"
	"unicode"
)

type TFIDFEmbedder struct {
	mu       sync.RWMutex
	docFreq  map[string]int
	docCount int
	dim      int
}

func NewTFIDFEmbedder(dim int) *TFIDFEmbedder {
	return &TFIDFEmbedder{
		docFreq: make(map[string]int),
		dim:     dim,
	}
}

func (e *TFIDFEmbedder) EmbedFunc() func(ctx context.Context, text string) ([]float64, error) {
	return func(ctx context.Context, text string) ([]float64, error) {
		return e.Embed(ctx, text)
	}
}

func (e *TFIDFEmbedder) Embed(_ context.Context, text string) ([]float64, error) {
	tokens := tokenize(text)
	if len(tokens) == 0 {
		return make([]float64, e.dim), nil
	}

	// Count term frequency
	tf := make(map[string]int)
	uniqueTokens := make(map[string]struct{})
	for _, t := range tokens {
		tf[t]++
		uniqueTokens[t] = struct{}{}
	}

	// Update document frequency
	e.mu.Lock()
	e.docCount++
	for t := range uniqueTokens {
		e.docFreq[t]++
	}
	e.mu.Unlock()

	// Build vector using hashing trick
	vec := make([]float64, e.dim)
	e.mu.RLock()
	docCount := e.docCount
	for term, count := range tf {
		tfVal := float64(count) / float64(len(tokens))
		df := e.docFreq[term]
		idf := math.Log(float64(docCount+1)/float64(df+1)) + 1.0
		idx := hashToIndex(term, e.dim)
		vec[idx] += tfVal * idf
	}
	e.mu.RUnlock()

	// L2 normalize
	normalize(vec)
	return vec, nil
}

func tokenize(text string) []string {
	text = strings.ToLower(text)
	var tokens []string
	var current strings.Builder

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			current.WriteRune(r)
		} else {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
		}
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}
	return tokens
}

func hashToIndex(s string, dim int) int {
	h := fnv.New32a()
	h.Write([]byte(s))
	return int(h.Sum32()) % dim
}

func normalize(vec []float64) {
	var norm float64
	for _, v := range vec {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return
	}
	for i := range vec {
		vec[i] /= norm
	}
}
