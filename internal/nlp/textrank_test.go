package nlp

import (
	"strings"
	"testing"
	"time"
)

// --- 基本機能テスト ---

func TestExtractSummary_Basic(t *testing.T) {
	text := "The cat sat on the mat. The dog chased the cat. The bird flew over the tree. The fish swam in the pond. The cat climbed the tree."
	result := ExtractSummary(text, 2)
	if result == "" {
		t.Fatal("expected non-empty summary")
	}
	if result == text {
		t.Error("expected summary to be shorter than original")
	}
}

func TestExtractSummary_Japanese(t *testing.T) {
	text := "猫がマットの上に座った。犬が猫を追いかけた。鳥が木の上を飛んだ。魚が池を泳いだ。猫が木に登った。"
	result := ExtractSummary(text, 2)
	if result == "" {
		t.Fatal("expected non-empty summary for Japanese text")
	}
	if strings.Count(result, "。") > 2 {
		t.Errorf("expected at most 2 sentences, got: %s", result)
	}
}

func TestExtractSummary_FewerSentencesThanMax(t *testing.T) {
	text := "Only one sentence here."
	result := ExtractSummary(text, 3)
	if result != text {
		t.Errorf("expected original text when fewer sentences than max, got: %q", result)
	}
}

func TestExtractSummary_DefaultMaxSentences(t *testing.T) {
	text := "A. B. C. D. E. F."
	result := ExtractSummary(text, 0)
	if result == "" {
		t.Fatal("expected non-empty result with default maxSentences")
	}
}

func TestExtractSummary_PreservesOrder(t *testing.T) {
	text := "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here."
	result := ExtractSummary(text, 3)
	sentences := splitSentences(result)
	for i := 1; i < len(sentences); i++ {
		idxPrev := strings.Index(text, sentences[i-1])
		idxCurr := strings.Index(text, sentences[i])
		if idxPrev >= idxCurr {
			t.Errorf("sentences not in original order: %q before %q", sentences[i-1], sentences[i])
		}
	}
}

// --- セキュリティテスト ---

func TestExtractSummary_EmptyInput(t *testing.T) {
	result := ExtractSummary("", 3)
	if result != "" {
		t.Errorf("expected empty result for empty input, got: %q", result)
	}
}

func TestExtractSummary_WhitespaceOnly(t *testing.T) {
	result := ExtractSummary("   \t\n  ", 3)
	if result != "   \t\n  " && result != "" {
		t.Errorf("unexpected result for whitespace input: %q", result)
	}
}

func TestExtractSummary_NegativeMaxSentences(t *testing.T) {
	text := "A. B. C. D. E."
	result := ExtractSummary(text, -1)
	if result == "" {
		t.Fatal("expected non-empty result with negative maxSentences")
	}
}

func TestExtractSummary_LargeInput_NoResourceExhaustion(t *testing.T) {
	// 大量の文を入力してもメモリ・時間が爆発しないことを確認
	var sb strings.Builder
	for i := 0; i < 1000; i++ {
		sb.WriteString("This is sentence number ")
		sb.WriteString(strings.Repeat("word ", 10))
		sb.WriteString(". ")
	}
	text := sb.String()

	start := time.Now()
	result := ExtractSummary(text, 3)
	elapsed := time.Since(start)

	if result == "" {
		t.Fatal("expected non-empty result for large input")
	}
	if elapsed > 10*time.Second {
		t.Errorf("ExtractSummary took too long for 1000 sentences: %v", elapsed)
	}
}

func TestExtractSummary_VeryLongSingleSentence(t *testing.T) {
	// 文末記号なしの巨大な1文: パニックしないこと
	text := strings.Repeat("word ", 100000)
	result := ExtractSummary(text, 3)
	if result != text {
		t.Error("expected original text returned when only 1 sentence")
	}
}

func TestExtractSummary_RepeatedIdenticalSentences(t *testing.T) {
	// 全て同じ文 → 類似度行列が均一でもパニックしない
	text := strings.Repeat("The same sentence here. ", 20)
	result := ExtractSummary(text, 3)
	if result == "" {
		t.Fatal("expected non-empty result for repeated sentences")
	}
}

func TestExtractSummary_NoCommonTokens(t *testing.T) {
	// 全く共通トークンがない文 → 類似度0でもパニックしない
	text := "alpha beta gamma. delta epsilon zeta. eta theta iota. kappa lambda mu. nu xi omicron."
	result := ExtractSummary(text, 2)
	if result == "" {
		t.Fatal("expected non-empty result")
	}
}

func TestExtractSummary_SpecialCharacters(t *testing.T) {
	// 特殊文字・制御文字が含まれてもクラッシュしない
	text := "Hello <script>alert('xss')</script>. Normal sentence here. Another with $pecial ch@rs!. End of text。"
	result := ExtractSummary(text, 2)
	if result == "" {
		t.Fatal("expected non-empty result with special characters")
	}
}

func TestExtractSummary_NullBytes(t *testing.T) {
	text := "First sentence.\x00Second sentence.\x00Third sentence.\x00Fourth sentence."
	result := ExtractSummary(text, 2)
	if result == "" {
		t.Fatal("expected non-empty result with null bytes")
	}
}

func TestExtractSummary_UnicodeEdgeCases(t *testing.T) {
	// 絵文字、サロゲートペア、結合文字
	text := "Hello 🌍 world. This is a test 👨‍👩‍👧‍👦. Some math: ∑∫∂. Another line here. Final sentence."
	result := ExtractSummary(text, 2)
	if result == "" {
		t.Fatal("expected non-empty result with unicode edge cases")
	}
}

func TestExtractSummary_MixedLineEndings(t *testing.T) {
	text := "Line one.\r\nLine two.\rLine three.\nLine four.\r\nLine five."
	result := ExtractSummary(text, 2)
	if result == "" {
		t.Fatal("expected non-empty result with mixed line endings")
	}
}

func TestExtractSummary_SQLInjectionPatterns(t *testing.T) {
	text := "SELECT * FROM users. DROP TABLE users; --. INSERT INTO data VALUES('test'). Normal sentence here. Another one."
	result := ExtractSummary(text, 2)
	if result == "" {
		t.Fatal("expected non-empty result with SQL patterns")
	}
}

func TestExtractSummary_PathTraversalPatterns(t *testing.T) {
	text := "File at ../../etc/passwd found. Config in /etc/shadow is readable. The path ../../../root/.ssh/id_rsa exists. Normal file here. End."
	result := ExtractSummary(text, 2)
	if result == "" {
		t.Fatal("expected non-empty result with path traversal patterns")
	}
}

// --- 内部関数テスト ---

func TestSplitSentences_MixedDelimiters(t *testing.T) {
	text := "English end. Japanese end。Exclaim! Question? Japanese exclaim！Japanese question？"
	sentences := splitSentences(text)
	if len(sentences) != 6 {
		t.Errorf("expected 6 sentences, got %d: %v", len(sentences), sentences)
	}
}

func TestTokenize_EmptyInput(t *testing.T) {
	tokens := tokenize("")
	if len(tokens) != 0 {
		t.Errorf("expected 0 tokens for empty input, got %d", len(tokens))
	}
}

func TestCosineSimilarity_EmptyVectors(t *testing.T) {
	a := map[string]float64{}
	b := map[string]float64{}
	sim := cosineSimilarity(a, b)
	if sim != 0 {
		t.Errorf("expected 0 similarity for empty vectors, got %f", sim)
	}
}

func TestCosineSimilarity_Orthogonal(t *testing.T) {
	a := map[string]float64{"x": 1.0}
	b := map[string]float64{"y": 1.0}
	sim := cosineSimilarity(a, b)
	if sim != 0 {
		t.Errorf("expected 0 for orthogonal vectors, got %f", sim)
	}
}

func TestCosineSimilarity_Identical(t *testing.T) {
	a := map[string]float64{"x": 1.0, "y": 2.0}
	sim := cosineSimilarity(a, a)
	if sim < 0.999 || sim > 1.001 {
		t.Errorf("expected ~1.0 for identical vectors, got %f", sim)
	}
}

func TestTextRank_SingleNode(t *testing.T) {
	sim := [][]float64{{0}}
	scores := textRank(sim, 0.85, 30)
	if len(scores) != 1 {
		t.Fatalf("expected 1 score, got %d", len(scores))
	}
}

func TestTextRank_ZeroSimilarity(t *testing.T) {
	sim := [][]float64{
		{0, 0, 0},
		{0, 0, 0},
		{0, 0, 0},
	}
	scores := textRank(sim, 0.85, 30)
	for i, s := range scores {
		if s < 0 {
			t.Errorf("score[%d] = %f, expected non-negative", i, s)
		}
	}
}
