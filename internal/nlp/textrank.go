package nlp

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

// ExtractSummary performs extractive summarization using TextRank.
// It splits text into sentences, ranks them by importance via a PageRank-like
// algorithm over TF-IDF cosine similarity, and returns the top sentences
// in their original order.
// maxSentences limits how many sentences to keep. If <= 0, defaults to 3.
func ExtractSummary(text string, maxSentences int) string {
	if maxSentences <= 0 {
		maxSentences = 3
	}

	sentences := splitSentences(text)
	if len(sentences) <= maxSentences {
		return text
	}

	// Tokenize each sentence
	tokenized := make([][]string, len(sentences))
	for i, s := range sentences {
		tokenized[i] = tokenize(s)
	}

	// Build TF-IDF vectors for each sentence
	// First compute document frequency (sentence = document)
	df := make(map[string]int)
	for _, tokens := range tokenized {
		seen := make(map[string]bool)
		for _, t := range tokens {
			if !seen[t] {
				df[t]++
				seen[t] = true
			}
		}
	}
	n := len(sentences)

	type vec = map[string]float64
	vectors := make([]vec, n)
	for i, tokens := range tokenized {
		tf := make(map[string]int)
		for _, t := range tokens {
			tf[t]++
		}
		v := make(vec)
		for term, count := range tf {
			tfVal := float64(count) / float64(max(len(tokens), 1))
			idf := math.Log(float64(n+1)/float64(df[term]+1)) + 1.0
			v[term] = tfVal * idf
		}
		vectors[i] = v
	}

	// Build similarity matrix
	sim := make([][]float64, n)
	for i := range sim {
		sim[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			s := cosineSimilarity(vectors[i], vectors[j])
			sim[i][j] = s
			sim[j][i] = s
		}
	}

	// PageRank iteration
	scores := textRank(sim, 0.85, 30)

	// Rank sentences by score, pick top ones
	type ranked struct {
		index int
		score float64
	}
	ranking := make([]ranked, n)
	for i, s := range scores {
		ranking[i] = ranked{i, s}
	}
	sort.Slice(ranking, func(a, b int) bool {
		return ranking[a].score > ranking[b].score
	})

	// Select top sentences, preserve original order
	selected := make([]int, 0, maxSentences)
	for i := 0; i < maxSentences && i < len(ranking); i++ {
		selected = append(selected, ranking[i].index)
	}
	sort.Ints(selected)

	var parts []string
	for _, idx := range selected {
		parts = append(parts, sentences[idx])
	}
	return strings.Join(parts, " ")
}

// textRank runs the PageRank algorithm on a similarity matrix.
func textRank(sim [][]float64, damping float64, iterations int) []float64 {
	n := len(sim)
	scores := make([]float64, n)
	for i := range scores {
		scores[i] = 1.0
	}

	// Precompute row sums for normalization
	rowSum := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			rowSum[i] += sim[i][j]
		}
	}

	for iter := 0; iter < iterations; iter++ {
		newScores := make([]float64, n)
		for i := 0; i < n; i++ {
			sum := 0.0
			for j := 0; j < n; j++ {
				if i == j || rowSum[j] == 0 {
					continue
				}
				sum += sim[j][i] / rowSum[j] * scores[j]
			}
			newScores[i] = (1 - damping) + damping*sum
		}
		scores = newScores
	}
	return scores
}

func cosineSimilarity(a, b map[string]float64) float64 {
	dot := 0.0
	for k, v := range a {
		if bv, ok := b[k]; ok {
			dot += v * bv
		}
	}
	if dot == 0 {
		return 0
	}
	normA := 0.0
	for _, v := range a {
		normA += v * v
	}
	normB := 0.0
	for _, v := range b {
		normB += v * v
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// splitSentences splits text into sentences using common delimiters.
// Handles Japanese (。！？) and English (.!?) sentence endings.
func splitSentences(text string) []string {
	var sentences []string
	var current strings.Builder

	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		current.WriteRune(runes[i])
		if isSentenceEnd(runes[i]) {
			s := strings.TrimSpace(current.String())
			if s != "" {
				sentences = append(sentences, s)
			}
			current.Reset()
		}
	}
	// Remaining text as final sentence
	if s := strings.TrimSpace(current.String()); s != "" {
		sentences = append(sentences, s)
	}

	return sentences
}

func isSentenceEnd(r rune) bool {
	return r == '.' || r == '!' || r == '?' || r == '。' || r == '！' || r == '？'
}

// tokenize splits text into lowercase word tokens.
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
