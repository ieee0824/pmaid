# pmaid

Programming AI Assistant with Memory

プログラミングAIアシスタント。[memAI-go](https://github.com/ieee0824/memAI-go) v1.2.0 による短期・長期記憶システムを搭載し、会話を跨いだコンテキスト保持が可能です。

## Features

- 汎用LLMインターフェース（OpenAI / Google AI / ローカルサーバー対応）
- memAI-goによるメモリシステム
  - STM（短期記憶）: 活性化減衰モデルによるワーキングメモリ
  - LTM（長期記憶）: ベクトル検索 + 感情プライミング
  - 感情分析・フィードバック検出
- ファイル読み書き・コマンド実行ツール
- プランモード（大規模変更の計画→承認→実行フロー）
- 軽量LLMによるコンテキスト圧縮（Gemma等で要約生成）
- コンテキスト管理（自動圧縮 + イテレーション制限のラップアップヒント）
- カスタムスキル（`~/.pmaid/skills/` にプロンプトテンプレートを配置）
- 会話履歴の検索・閲覧（`pmaid history`）
- 対話型CLI / 直接クエリモード
- SQLiteによるメモリ永続化（ピュアGo、CGO不要）
- ローカルTF-IDF Embedding（外部API不要）
- エクスポネンシャルバックオフによるAPIリトライ（429/5xx対応）

## Installation

```bash
go install github.com/ieee0824/pmaid/cmd/pmaid@latest
```

## Usage

```bash
# 対話モード
pmaid

# 直接質問
pmaid -q "Hello, how can you help me?"

# 特定のディレクトリをコンテキストとして使用
pmaid --context ./src

# モデル指定
pmaid --model gpt-4o-mini

# 会話履歴を表示
pmaid history

# 最近50件の履歴を表示
pmaid history -n 50

# キーワードで履歴を検索
pmaid history "リファクタリング"
```

## Configuration

### 初回セットアップ

設定ファイルがない状態で起動すると、対話的にセットアップが実行されます。

### 設定ファイル

`~/.pmaid/config.toml` に配置（`--config` フラグでパス変更可能）。全項目にデフォルト値があるため、変更したい項目のみ記述すれば動作します。

```toml
# トップレベル設定
name = "pmaid"             # AIアシスタントの名前
model = "gpt-4o"
context_dir = "."
memory_path = ""           # 空の場合 ~/.pmaid/memory

# メインLLM設定
[llm]
provider = "openai"        # "openai", "google", "local"
api_key = ""               # 環境変数 OPENAI_API_KEY でも可
model = "gpt-4o"
base_url = ""              # カスタムエンドポイント

# 軽量LLM設定 (コンテキスト圧縮の要約生成に使用、省略可)
[light_llm]
provider = "google"        # "google", "openai", "local"
api_key = ""               # 環境変数 GEMINI_API_KEY でも可
model = "gemma-3-4b-it"
base_url = ""

# ローカルサーバーの場合 (Ollama等)
# [light_llm]
# provider = "local"
# model = "gemma3:4b"
# base_url = "http://localhost:11434/v1/"

[memory]
embedding_dim = 512

[stm]
max_items = 7
activation_threshold = 0.1
normal_decay_rate = 0.15
emotional_decay_rate = 0.07
refresh_boost = 0.3

[ltm]
top_k = 5
similarity_threshold = 0.3
thread_boost = 0.1
date_boost = 0.15
emotional_boost = 0.12

[agent]
max_tool_iterations = 50   # 1ターンあたりのツール呼び出し上限
max_context_chars = 100000 # コンテキスト圧縮が発動する文字数上限
```

### 環境変数

```bash
export OPENAI_API_KEY="your-api-key"
export GEMINI_API_KEY="your-gemini-key"       # Google AI / Gemma
export PMAID_MEMORY_PATH="~/.pmaid/memory"    # 設定ファイルより優先

# カラー出力
# - デフォルト: TTYのときのみ有効（パイプ/リダイレクト時は無効）
export NO_COLOR=1                 # カラー出力を無効化
export PMAID_FORCE_COLOR=1         # TTY判定等に関わらず強制的に有効化
```

### 優先順位

CLIフラグ > 環境変数 > 設定ファイル > デフォルト値

## Architecture

```
cmd/pmaid/main.go              # CLIエントリポイント
internal/
  config/
    config.go                  # TOML設定ファイル読み込み + 対話的セットアップ
  llm/
    llm.go                     # LLMインターフェース定義
    openai/openai.go           # OpenAI実装 (リトライ付き)
    google/google.go           # Google AI実装 (Gemini API互換エンドポイント)
  memory/
    embedding.go               # TF-IDF ローカルEmbedding (512次元)
    store.go                   # SQLite MemoryStore + 履歴検索
  tools/
    tools.go                   # Toolインターフェース + Registry (ファジーマッチ)
    fileread.go                # read_file
    filewrite.go               # write_file
    exec.go                    # execute_command
    plan.go                    # create_plan / update_plan_step / show_plan
  agent/
    agent.go                   # エージェントコアループ + コンテキスト圧縮
  skills/
    skills.go                  # スキルテンプレート読み込み
  logger/
    logger.go                  # 日別ログファイル出力
  spinner/
    spinner.go                 # CLI用アニメーション表示
```

### メモリシステムの動作

1. **感情分析**: ユーザー入力から感情状態を検出（日本語/英語キーワードベース）
2. **フィードバック検出**: ポジティブ/ネガティブなフィードバックを検出し、関連記憶のブースト値を調整
3. **LTM検索**: ベクトル類似度 + 感情プライミングで関連する過去の記憶を検索
4. **STM更新**: ワーキングメモリを活性化減衰モデルで管理
5. **LLMツールループ**: システムプロンプトに記憶コンテキストを注入し、ツール呼び出しを実行（コンテキスト上限超過時は自動圧縮）
6. **LTM保存**: 会話をEmbedding化して長期記憶に保存

### コンテキスト圧縮

メッセージが `max_context_chars` を超えると自動的に圧縮が発動します。

- **軽量LLM設定時**: Gemma等で古いツール結果やアシスタント応答を1-2文に要約
- **未設定時**: 古いツール結果をプレースホルダーに置換、長い応答を先頭200文字に切り捨て

残り3イテレーションでラップアップヒントが注入され、LLMに作業のまとめを促します。

## License

MIT
