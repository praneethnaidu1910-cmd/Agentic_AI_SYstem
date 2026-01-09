# SWE645 - Agentic AI System

Team Members:
Praneeth Naidu,  
Ganesh Jasti, 
Venkata Abhiram Karuturi, 
Jithendra Sai Pappuri


## Source Code

The complete source code for this project is available on GitHub:  
🔗 **Repository:** https://github.com/praneethnaidu1910-cmd/Agentic_AI_SYstem/tree/main


## Overview

This project implements an agentic AI system using LangGraph that demonstrates the core components of modern AI agents:
- **LLM** (Ollama/Gemini/Hugging Face) for reasoning and planning
- **Tools** for acting and integration (web search, calculator, file operations)
- **Short-term memory** (conversation history)
- **Long-term memory** (FAISS vector store)
- **Human-in-the-loop** feedback mechanism

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Agentic AI System                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │     LLM      │───▶│   Planner    │───▶│   Executor   │  │
│  │  (OpenAI)    │    │  (LangGraph) │    │   (Tools)    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │          │
│         └────────────────────┼────────────────────┘          │
│                              │                               │
│  ┌───────────────────────────┼───────────────────────────┐  │
│  │                    State Manager                        │  │
│  │  ┌──────────────┐              ┌──────────────┐        │  │
│  │  │ Short-term   │              │ Long-term    │        │  │
│  │  │   Memory     │              │   Memory     │        │  │
│  │  │(Conversation)│              │ (FAISS DB)   │        │  │
│  │  └──────────────┘              └──────────────┘        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                              │                               │
│  ┌───────────────────────────┴───────────────────────────┐  │
│  │              Human-in-the-Loop                        │  │
│  │         (Approval for Critical Actions)               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- LLM provider (choose one):
  - **Ollama** (Recommended - 100% free, local): Install from https://ollama.ai
  - **Google Gemini** (Free tier): Get API key from https://aistudio.google.com/app/apikey
  - **Hugging Face** (Free with token): Get token from https://huggingface.co/settings/tokens

### Installation

1. Clone this repository or extract the project files.

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your LLM provider:

   **Option 1: Ollama (Recommended - No API key needed)**
   ```bash
   # Install Ollama from https://ollama.ai
   # Pull a model:
   ollama pull llama3.2
   ```

   **Option 2: Google Gemini**
   Create a `.env` file:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```

   **Option 3: Hugging Face**
   Create a `.env` file:
   ```
   HUGGINGFACE_API_KEY=your_huggingface_token_here
   ```

4. (Optional) For web search functionality, add to `.env`:
```
SERPAPI_API_KEY=your_serpapi_key_here
```

### Running the System

#### Basic Example
```bash
python main.py
```

#### Interactive Mode
```bash
python interactive_demo.py
```

## Project Structure

```
SWE645_EXTRA_CREDIT/
├── agentic_system/
│   ├── __init__.py
│   ├── agent.py              # Main agent implementation
│   ├── tools.py              # Tool definitions
│   ├── memory.py             # Memory management (short & long-term)
│   └── state.py              # State definitions
├── main.py                   # Entry point
├── interactive_demo.py       # Interactive demo script
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── .env                      # Environment variables (create this)
```

## Components Explained

### 1. LLM (Large Language Model)
- **Purpose**: Reasoning and planning
- **Implementation**: Ollama (Llama 3.2), Google Gemini, or Hugging Face via LangChain
- **Role**: Analyzes user queries, plans actions, decides which tools to use

### 2. Tools / MCP (Model Context Protocol)
- **Purpose**: Acting and integration
- **Tools Implemented**:
  - `WebSearchTool`: Search the web for information
  - `CalculatorTool`: Perform mathematical calculations
  - `FileReadTool`: Read files from disk
  - `FileWriteTool`: Write files to disk (requires approval)

### 3. Short-term Memory
- **Purpose**: Maintain conversation context
- **Implementation**: Conversation history stored in agent state
- **Role**: Allows the agent to reference previous interactions

### 4. Long-term Memory
- **Purpose**: Persistent knowledge storage
- **Implementation**: FAISS vector database
- **Role**: Stores and retrieves information across sessions

### 5. Human-in-the-Loop (HITL)
- **Purpose**: Oversight and feedback
- **Implementation**: Approval prompts for critical actions (file writes, deletions)
- **Role**: Ensures user control over potentially destructive operations

## Example Usage

```python
from agentic_system import AgenticSystem

# Initialize the agent (uses Ollama by default)
agent = AgenticSystem(provider="ollama", model_name="llama3.2")

# Or use Gemini
# agent = AgenticSystem(provider="gemini")

# Run a query
response = agent.run("What is the capital of France? Then calculate 15 * 23.")
print(response)
```

## Use Case

This agentic system serves as an **Intelligent Research Assistant** that can:
- Answer questions using web search
- Perform calculations
- Store and retrieve information from memory
- Read and write files (with approval)
- Maintain context across conversations

## Technical Details

- **Framework**: LangGraph for stateful agent orchestration
- **LLM**: Ollama (Llama 3.2), Google Gemini, or Hugging Face (configurable)
- **Vector Store**: FAISS for efficient similarity search
- **Embeddings**: Sentence-transformers (free, local) or OpenAI embeddings
- **State Management**: LangGraph StateGraph with typed state

## Challenges and Solutions

1. **State Management**: Used LangGraph's StateGraph for clean state transitions
2. **Memory Integration**: Combined conversation history with vector embeddings
3. **Tool Orchestration**: Implemented tool selection logic based on query analysis
4. **Human Feedback**: Added approval gates for critical operations

## Future Enhancements

- Multi-agent collaboration
- More sophisticated planning strategies
- Integration with more external APIs
- Enhanced memory retrieval strategies

## License

This project is created for educational purposes as part of SWE645 coursework.

