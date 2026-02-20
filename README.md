# journal-ai

A CLI tool that uses AI (local or cloud LLM) to structure journal entries and save them via file-journal.

## How it works

```
Input → journal-ai → LLM (Ollama/OpenAI) → structured entry → file-journal → saved
```

## Installation

### Prerequisites
- [file-journal](https://github.com/total70/file-journal) must be installed
- For local AI: [Ollama](https://ollama.com) with `llama3.2` model
- For cloud AI: OpenAI API key (optional fallback)

### Build from source
```bash
git clone <repo-url>
cd journal-ai
cargo build --release
# Binary will be at: target/release/journal-ai
```

## Configuration

Create `~/.config/journal-ai/config.toml`:

```toml
[llm]
provider = "ollama"  # or "openai"

[ollama]
base_url = "http://localhost:11434"
model = "llama3.2"   # or "llama3.2:3b", "gemma2:2b"

[openai]
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
# API key from OPENAI_API_KEY env var (recommended)
```

Or run interactive setup:
```bash
journal-ai init
```

## Usage

### Basic usage
```bash
# Create a structured journal entry
journal-ai "Met with team to discuss Q1 planning"

# From stdin
echo "Ideas for new project" | journal-ai

# With specific provider
journal-ai --provider openai "Important meeting notes"

# Preview before saving
journal-ai --preview "Test entry"

# Dry run (don't save)
journal-ai --dry-run "Test entry"
```

### Check setup
```bash
journal-ai doctor
```

## Features

- **Multiple LLM providers**: Ollama (local, default) or OpenAI (cloud)
- **Automatic structuring**: AI generates title, content, and tags
- **File-journal integration**: Seamlessly saves to your journal
- **Configurable**: TOML config + environment variables
- **Fast**: Optimized for small models (3B parameters)

## Recommended Models

| Model | Params | Speed | Quality |
|-------|--------|-------|---------|
| llama3.2 | 3B | Fast | Excellent |
| llama3.2:1b | 1B | Very fast | Good |
| gemma2:2b | 2B | Very fast | Good |
| gpt-4o-mini | cloud | Instant | Excellent |

## Testing

```bash
cargo test
```

## License

MIT
