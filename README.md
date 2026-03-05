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

環境変数で設定：

```bash
export OPENAI_API_KEY="your-api-key"
export PMAID_MEMORY_PATH="~/.pmaid/memory"  # デフォルト: ~/.pmaid/memory
```

## Architecture

```
cmd/pmaid/main.go              # CLIエントリポイント
internal/
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
