# pmaid

Programming AI Assistant with Memory

Claude Codeのようなプログラミングエージェントで、memAI-goを使用した高度なメモリシステムを搭載しています。

## Features

- 🤖 Claude APIを使用したAIアシスタント
- 🧠 memAI-goによる長期記憶システム
- 📝 ファイル操作機能
- ⚡ コマンド実行機能
- 💬 対話型CLI

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
```

## Configuration

環境変数で設定：

```bash
export ANTHROPIC_API_KEY="your-api-key"
export PMAID_MEMORY_PATH="~/.pmaid/memory"
```
