# Agentic AI System

A production-style agentic AI framework built with LangGraph that demonstrates core components of autonomous AI agents: reasoning, tool use, memory, and human oversight.

## What It Does

- **LLM-powered reasoning** — Supports Ollama (local), Google Gemini, and Hugging Face models via LangChain
- **Tool execution** — Web search, calculator, and file operations via a modular tool interface
- **Short-term memory** — Conversation history maintained across agent turns
- **Long-term memory** — FAISS vector store for persistent knowledge retrieval across sessions
- **Human-in-the-loop control** — Approval gates for critical actions (file writes, deletions) before execution

## Architecture

```
┌─────────────────────────────────────────┐
│             Agentic AI System            │
│                                          │
│  LLM ──► Planner ──► Executor (Tools)   │
│               │                          │
│         State Manager                    │
│    Short-term    Long-term (FAISS)       │
│               │                          │
│      Human-in-the-Loop Gate              │
└─────────────────────────────────────────┘
```

## Tech Stack

- **Orchestration**: LangGraph (stateful agent graph)
- **LLM**: Ollama / Google Gemini / Hugging Face (configurable)
- **Vector Store**: FAISS with sentence-transformers embeddings
- **Language**: Python 3.9+

## Quick Start

```bash
git clone https://github.com/praneethnaidu1910-cmd/Agentic_AI_SYstem
cd Agentic_AI_SYstem
pip install -r requirements.txt
```

Set up your LLM provider in a `.env` file:

```
# Option 1: Google Gemini (free tier)
GOOGLE_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash

# Option 2: Hugging Face
HUGGINGFACE_API_KEY=your_token_here

# Optional: Web search
SERPAPI_API_KEY=your_key_here
```

Or use Ollama locally (no API key needed):
```bash
ollama pull llama3.2
python main.py
```

## Usage

```python
from agentic_system import AgenticSystem

agent = AgenticSystem(provider="ollama", model_name="llama3.2")
response = agent.run("Search the web for latest AI security research and summarize it.")
print(response)
```

## Key Design Decisions

- **Why LangGraph?** Enables stateful, cyclical agent graphs — agents can loop, retry, and branch based on tool results
- **Why FAISS?** Lightweight vector store that runs locally without external dependencies
- **Why HITL gates?** Prevents autonomous file writes/deletes without explicit user approval — critical for safe agent deployment

## Future Work

- Multi-agent collaboration with shared memory
- Integration with security tooling APIs (Shodan, VirusTotal)
- Web-based dashboard for real-time agent action monitoring
