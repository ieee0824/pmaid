# pmaid

Programming AI Assistant with Memory

プログラミングAIアシスタント。[memAI-go](https://github.com/ieee0824/memAI-go) v1.2.0 による短期・長期記憶システムを搭載し、会話を跨いだコンテキスト保持が可能です。

## Features

- 🤖 汎用LLMインターフェース（OpenAI API対応、差し替え可能）
- 🧠 memAI-goによるメモリシステム
  - STM（短期記憶）: 活性化減衰モデルによるワーキングメモリ
  - LTM（長期記憶）: ベクトル検索 + 感情プライミング
  - 感情分析・フィードバック検出
- 📝 ファイル読み書きツール
- ⚡ コマンド実行ツール
- 💬 対話型CLI
- 💾 SQLiteによるメモリ永続化（ピュアGo、CGO不要）
- 🔤 ローカルTF-IDF Embedding（外部API不要）

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
```

## Configuration

### 設定ファイル

`~/.pmaid/config.toml` に設定ファイルを配置できます（`--config` フラグでパス変更可能）。

```toml
model = "gpt-4o"
context_dir = "."
memory_path = "~/.pmaid/memory"

[llm]
provider = "openai"
api_key = ""           # 環境変数 OPENAI_API_KEY でも可
model = "gpt-4o"

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
```

全項目にデフォルト値があるため、変更したい項目のみ記述すれば動作します。

### 環境変数

```bash
export OPENAI_API_KEY="your-api-key"
export PMAID_MEMORY_PATH="~/.pmaid/memory"  # 設定ファイルより優先
```

### 優先順位

CLIフラグ > 環境変数 > 設定ファイル > デフォルト値

## Architecture

```
cmd/pmaid/main.go              # CLIエントリポイント
internal/
  config/
    config.go                  # TOML設定ファイル読み込み
  llm/
    llm.go                     # LLMインターフェース定義
    openai/openai.go           # OpenAI実装
  memory/
    embedding.go               # TF-IDF ローカルEmbedding (512次元)
    store.go                   # SQLite MemoryStore
  tools/
    tools.go                   # Toolインターフェース + Registry
    fileread.go                # read_file
    filewrite.go               # write_file
    exec.go                    # execute_command
  agent/
    agent.go                   # エージェントコアループ
```

### メモリシステムの動作

1. **感情分析**: ユーザー入力から感情状態を検出（日本語/英語キーワードベース）
2. **フィードバック検出**: ポジティブ/ネガティブなフィードバックを検出し、関連記憶のブースト値を調整
3. **LTM検索**: ベクトル類似度 + 感情プライミングで関連する過去の記憶を検索
4. **STM更新**: ワーキングメモリを活性化減衰モデルで管理
5. **LLMツールループ**: システムプロンプトに記憶コンテキストを注入し、ツール呼び出しを最大10回実行
6. **LTM保存**: 会話をEmbedding化して長期記憶に保存

## License

MIT
