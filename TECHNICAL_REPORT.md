# SWE645 - Agentic AI System: Technical Report

**Course:** SWE645  
**Assignment:** Extra Credit - Building an Agentic AI System  
**Date:**  09 December 2025


Team Members:
Praneeth Naidu(G01477360)  
Ganesh Jasti(G01505410)  
Venkata Abhiram Karuturi(G01505660)  
Jithendra Sai Pappuri(G01506453) 

---

## 1. Introduction

AI Agents represent a new generation of intelligent systems that can reason, plan, and act autonomously. Unlike traditional AI systems that operate in isolation with fixed inputs and outputs, agentic systems combine Large Language Models (LLMs) for cognitive reasoning, tools and APIs for action execution, and memory systems for context retention. This project implements a fully functional agentic AI system using LangGraph, demonstrating how these components work together to create autonomous, goal-oriented behavior.

### 1.1 What Makes AI Agents Different from Traditional AI Systems

Traditional AI systems are typically:
- **Reactive**: Respond to specific inputs with predetermined outputs
- **Stateless**: Each interaction is independent, with no memory of previous interactions
- **Single-purpose**: Designed for one specific task
- **Passive**: Require explicit instructions for every action

AI Agents, in contrast, are:
- **Proactive**: Can plan multi-step actions to achieve goals
- **Stateful**: Maintain both short-term (conversation) and long-term (knowledge) memory
- **Multi-capable**: Can use various tools and adapt to different tasks
- **Autonomous**: Make decisions about which tools to use and when to use them
- **Goal-oriented**: Work towards achieving objectives through reasoning and planning

The key difference is that agents can **reason about what actions to take**, **remember past interactions**, and **adapt their behavior** based on context, making them more like intelligent assistants than simple programs.

---

## 2. System Architecture

The agentic system is built using LangGraph, a graph-based orchestration framework that enables stateful, multi-step workflows. The architecture follows a pipeline pattern where each node represents a distinct phase of processing.

```
┌─────────────────────────────────────────────────────────────┐
│                    Agentic AI System                         │
│                    (LangGraph Workflow)                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  User Query                                                   │
│      ↓                                                        │
│  ┌──────────────────┐                                         │
│  │ Retrieve Memory │  ← Long-term Memory (FAISS)            │
│  │   (Node 1)      │                                         │
│  └────────┬─────────┘                                         │
│           ↓                                                    │
│  ┌──────────────────┐                                         │
│  │  Agent Node     │  ← LLM Reasoning & Planning            │
│  │   (Node 2)      │     (Ollama/Gemini/HuggingFace)        │
│  └────────┬─────────┘                                         │
│           ↓                                                    │
│  ┌──────────────────┐                                         │
│  │  Tools Node      │  ← Tool Execution                      │
│  │   (Node 3)       │     (Calculator, Search, Files)         │
│  └────────┬─────────┘                                         │
│           ↓                                                    │
│  ┌──────────────────┐                                         │
│  │ Check Approval   │  ← Human-in-the-Loop                   │
│  │   (Node 4)       │     (For critical actions)              │
│  └────────┬─────────┘                                         │
│           ↓                                                    │
│  ┌──────────────────┐                                         │
│  │  Respond Node    │  ← Generate Final Answer               │
│  │   (Node 5)       │                                         │
│  └────────┬─────────┘                                         │
│           ↓                                                    │
│  ┌──────────────────┐                                         │
│  │ Store Memory     │  → Long-term Memory (FAISS)            │
│  │   (Node 6)       │                                         │
│  └────────┬─────────┘                                         │
│           ↓                                                    │
│  Final Response                                                │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              State Management (LangGraph)              │  │
│  │  • Messages (Short-term memory)                       │  │
│  │  • Tool Results                                        │  │
│  │  • Memory Context                                      │  │
│  │  • Approval Flags                                      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 Workflow Description

The system follows a six-node workflow:

1. **Retrieve Memory**: Queries the FAISS vector store for relevant past information
2. **Agent Node**: LLM analyzes the query, plans actions, and decides which tools to use
3. **Tools Node**: Executes selected tools (calculator, web search, file operations)
4. **Check Approval**: Verifies if any actions require human approval
5. **Respond Node**: Generates the final answer using tool results and context
6. **Store Memory**: Saves important information to long-term memory for future use

The workflow is stateful, meaning each node can access and modify the shared state, allowing information to flow between components seamlessly.

---

## 3. Component Explanations

### 3.1 LLM – Reasoning and Planning

**Purpose**: The Large Language Model serves as the "brain" of the agent, providing reasoning capabilities and decision-making.

**Implementation**: 
- **Primary**: Ollama with Llama 3.2 (100% free, runs locally)
- **Alternatives**: Google Gemini (free tier), Hugging Face (free with token)
- **Framework**: LangChain for LLM integration

**How it works**:
1. Receives user query and context (memory, conversation history)
2. Analyzes the query to understand intent
3. Plans which tools are needed to answer the query
4. Generates tool calls or direct responses
5. Synthesizes final answer from tool results

**Example**: When asked "Calculate 15 * 23", the LLM:
- Recognizes this is a calculation task
- Decides to use the calculator tool (if tool calling supported)
- Or performs the calculation directly
- Returns: "15 * 23 = 345"

**Key Feature**: The LLM uses a system prompt that instructs it on available tools and when to use them, enabling autonomous tool selection.

---

### 3.2 Tools / MCP – Acting and Integration

**Purpose**: Tools enable the agent to interact with the external world, perform actions, and access information beyond its training data.

**Implementation**: Four tools implemented using LangChain's `@tool` decorator:

1. **Web Search Tool** (`web_search_tool`)
   - Searches the web for current information
   - Uses SerpAPI (optional) or mock implementation
   - Returns search results as text

2. **Calculator Tool** (`calculator_tool`)
   - Performs mathematical calculations safely
   - Supports arithmetic operations, functions (sqrt, sin, cos, etc.)
   - Uses safe evaluation with restricted namespace

3. **File Read Tool** (`file_read_tool`)
   - Reads files from the local filesystem
   - Validates file existence and permissions
   - Returns file contents as text

4. **File Write Tool** (`file_write_tool`)
   - Writes content to files
   - **Requires human approval** (Human-in-the-Loop)
   - Creates directories if needed

**Tool Integration**:
- Tools are bound to the LLM using `llm.bind_tools()`
- LLM can automatically call tools when needed
- Tool results are fed back to the LLM for response generation
- Tool execution is logged in the agent state

**Example Flow**:
```
User: "What is 25 squared? Then search for AI agents."
  ↓
LLM decides: Use calculator_tool, then web_search_tool
  ↓
Tools execute: 
  - Calculator: 25² = 625
  - Web Search: Returns information about AI agents
  ↓
LLM synthesizes: "25 squared is 625. [AI agent information]"
```

---

### 3.3 Memory – Short-term vs. Long-term Context

**Purpose**: Memory enables the agent to maintain context across interactions and learn from past experiences.

#### Short-term Memory (Conversation History)

**Implementation**: 
- Stored in `AgentState.messages` as a list of message dictionaries
- Maintains last 5 messages for context
- Automatically included in each LLM call

**How it works**:
- Each user query and agent response is stored
- Previous conversation is formatted and added to LLM context
- Allows agent to reference earlier parts of the conversation

**Example**:
```
User: "What is the capital of France?"
Agent: "The capital of France is Paris."
User: "What is its population?"
Agent: [Uses short-term memory] "Paris has approximately 2.1 million people."
```

#### Long-term Memory (Vector Store)

**Implementation**:
- **Technology**: FAISS (Facebook AI Similarity Search) vector database
- **Embeddings**: Sentence-transformers (free, local) or OpenAI embeddings
- **Storage**: Persistent on disk in `./memory_store/`

**How it works**:
1. **Storage**: When important information is generated, it's converted to embeddings and stored in FAISS
2. **Retrieval**: When a new query arrives, the system:
   - Converts query to embedding
   - Searches FAISS for similar past information
   - Retrieves top 3 most relevant items
   - Includes them in the LLM context

**Example**:
```
Query: "What did we discuss about AI agents earlier?"
  ↓
System searches long-term memory
  ↓
Retrieves: "Query: What are AI agents? Results: [past discussion]"
  ↓
LLM uses this context to answer
```

**Benefits**:
- **Persistence**: Information survives across sessions
- **Semantic Search**: Finds relevant information even with different wording
- **Scalability**: Can store thousands of interactions efficiently

---

### 3.4 Human-in-the-Loop – Oversight and Feedback

**Purpose**: Human-in-the-loop (HITL) provides safety and control by requiring human approval for critical or potentially destructive actions.

**Implementation**:
- Approval mechanism for file write operations
- Approval flags in agent state (`requires_approval`, `approval_pending`)
- Separate node in workflow (`check_approval`) that intercepts actions requiring approval

**How it works**:
1. When a tool returns "APPROVAL_REQUIRED", the workflow routes to `check_approval` node
2. The system sets approval flags in state
3. In production, this would prompt the user for approval
4. For demo, actions are auto-approved but logged
5. Approved actions proceed; rejected actions are cancelled

**Example Flow**:
```
User: "Save this data to file.txt"
  ↓
Agent decides to use file_write_tool
  ↓
Tool returns: "APPROVAL_REQUIRED: Write to 'file.txt'"
  ↓
System routes to check_approval node
  ↓
[In production: User sees prompt, approves/rejects]
  ↓
If approved: File is written
If rejected: Action cancelled, user notified
```

**Benefits**:
- **Safety**: Prevents accidental file modifications
- **Control**: User maintains oversight of critical operations
- **Transparency**: User knows what actions the agent is taking

**Future Enhancement**: Could be extended to require approval for:
- Web searches (to control API usage)
- Large file operations
- Network requests
- Database modifications

---

## 4. Agent Use Case and Goals

### 4.1 Use Case: Intelligent Research Assistant

The agentic system serves as an **Intelligent Research Assistant** that can:

1. **Answer Questions**: Uses web search and knowledge to answer queries
2. **Perform Calculations**: Handles mathematical operations
3. **Manage Information**: Stores and retrieves knowledge from memory
4. **File Operations**: Reads and writes files (with approval)
5. **Multi-step Tasks**: Combines multiple tools to complete complex requests

### 4.2 Goals

**Primary Goals**:
- **Autonomy**: Make decisions about which tools to use without explicit instructions
- **Context Awareness**: Remember past interactions and use that context
- **Tool Integration**: Seamlessly combine multiple tools for complex tasks
- **Safety**: Require approval for potentially destructive operations
- **Accessibility**: Work with free, local LLM options (Ollama) to avoid costs

**Example Scenarios**:

**Scenario 1: Research Task**
```
User: "What is the capital of France? Then calculate the distance from Paris to London."
Agent:
  1. Uses web search or knowledge: "Paris is the capital"
  2. Uses calculator or knowledge: "Distance is approximately 344 km"
  3. Stores this information in long-term memory
```

**Scenario 2: Information Retrieval**
```
User: "What did we discuss about AI agents last time?"
Agent:
  1. Searches long-term memory for "AI agents"
  2. Retrieves relevant past conversation
  3. Summarizes: "We discussed that AI agents are autonomous systems..."
```

**Scenario 3: Multi-tool Task**
```
User: "Calculate 25 squared, then search for information about LangGraph."
Agent:
  1. Uses calculator_tool: 25² = 625
  2. Uses web_search_tool: Finds LangGraph information
  3. Synthesizes both results into coherent response
```

---

## 5. Example Run and Transcript

### 5.1 System Initialization

```
Initializing agentic system...
[OK] Using Ollama (llama3.2) - COMPLETELY FREE (local)
[OK] Agent initialized successfully
```

### 5.2 Example Conversation

**Transcript 1: Simple Calculation**

```
User: Calculate 15 * 23

Agent Processing:
  → Retrieve Memory: No relevant past information
  → Agent Node: Recognizes calculation task
  → Tools Node: (No tool calls, LLM calculates directly)
  → Respond Node: Generates answer
  → Store Memory: Saves query and result

Response: 15 * 23 = 345
```

**Transcript 2: Knowledge Question**

```
User: What is the capital of France?

Agent Processing:
  → Retrieve Memory: Checks for past discussions about France
  → Agent Node: Recognizes knowledge question
  → Tools Node: (No tool calls needed, uses LLM knowledge)
  → Respond Node: Generates answer
  → Store Memory: Saves question and answer

Response: The capital of France is Paris.
```

**Transcript 3: Complex Multi-tool Query**

```
User: What is 25 squared? Then search for information about AI agents.

Agent Processing:
  → Retrieve Memory: No relevant information
  → Agent Node: Plans to calculate, then search
  → Tools Node: 
      - Calculator: 25² = 625
      - Web Search: Retrieves AI agent information
  → Respond Node: Synthesizes both results
  → Store Memory: Saves query and comprehensive results

Response: 
25 squared (25²) equals 625.

Now, regarding AI agents:

An AI agent is an autonomous software program designed to make 
decisions based on its environment and interactions with users or 
other systems. AI agents can be found in various applications, such 
as chatbots, virtual assistants, and robotic systems.

[Detailed information about AI agents continues...]
```

### 5.3 System State During Execution

**State Snapshot (after Transcript 3)**:
```python
{
  "messages": [
    {"role": "user", "content": "What is 25 squared?..."},
    {"role": "assistant", "content": "25 squared (25²) equals 625..."}
  ],
  "current_query": "What is 25 squared? Then search for information about AI agents.",
  "tool_results": [
    {"tool": "calculator_tool", "result": "Result: 625"},
    {"tool": "web_search_tool", "result": "[Search results about AI agents]"}
  ],
  "memory_context": ["Query: What is 25 squared?...", "..."],
  "step_count": 3,
  "requires_approval": False
}
```

---

## 6. Challenges Faced and Lessons Learned

### 6.1 Challenges

#### Challenge 1: LLM Provider Selection and Quota Limits

**Problem**: Initially used Google Gemini, but hit strict free tier quotas (5 requests/minute for gemini-2.5-flash).

**Solution**: 
- Implemented multiple LLM providers (Ollama, Gemini, Hugging Face)
- Made Ollama the default (100% free, no quotas, runs locally)
- Created LLM factory pattern for easy provider switching

**Lesson**: Always have fallback options and consider free, local alternatives for development.

#### Challenge 2: Tool Calling with Local LLMs

**Problem**: Ollama models don't support structured tool calling like OpenAI/Gemini, making automatic tool selection difficult.

**Solution**:
- Implemented fallback logic for LLMs without tool calling support
- Enhanced prompts to guide LLM to use tools when needed
- Added manual tool execution based on LLM reasoning
- Used more capable models (llama3.2) that better understand tool usage

**Lesson**: Not all LLMs support the same features; need to adapt architecture for different providers.

#### Challenge 3: Response Extraction and Empty Responses

**Problem**: Some LLM calls returned empty content, causing the system to fail silently.

**Solution**:
- Added robust response extraction handling both string and message object responses
- Implemented fallback logic for empty responses
- Added direct calculation for math queries when LLM fails
- Simplified message formatting to improve LLM reliability

**Lesson**: Always handle edge cases and provide fallbacks for LLM failures.

#### Challenge 4: State Management in LangGraph

**Problem**: Managing state across multiple nodes while ensuring type safety and proper data flow.

**Solution**:
- Used TypedDict for type-safe state definition
- Implemented clear state structure with all required fields
- Used LangGraph's built-in state management features
- Added proper state updates in each node

**Lesson**: Type safety and clear state structure are crucial for complex workflows.

#### Challenge 5: Memory Integration

**Problem**: Integrating both short-term (conversation) and long-term (vector store) memory seamlessly.

**Solution**:
- Separated memory concerns into MemoryManager class
- Used FAISS for efficient vector storage and retrieval
- Implemented free embeddings (sentence-transformers) to avoid costs
- Made memory retrieval automatic at workflow start

**Lesson**: Modular design makes complex features manageable.

### 6.2 Lessons Learned

1. **Framework Choice Matters**: LangGraph's stateful workflow pattern is ideal for agentic systems, making it easier to manage complex multi-step processes.

2. **Free Options Are Viable**: Ollama provides a completely free, local alternative to paid APIs, perfect for development and testing.

3. **Tool Integration Requires Care**: Not all LLMs support structured tool calling; need to design for both scenarios.

4. **Memory is Essential**: Both short-term and long-term memory significantly improve agent capabilities and user experience.

5. **Error Handling is Critical**: LLMs can fail or return unexpected formats; robust error handling and fallbacks are necessary.

6. **Modular Design Pays Off**: Separating components (LLM, tools, memory) makes the system maintainable and extensible.

7. **Documentation is Important**: Well-documented code and architecture diagrams help understand and present the system.

---

## 7. Conclusion

This project successfully demonstrates the core components of an agentic AI system:
- **LLM** for reasoning and planning
- **Tools** for action execution
- **Memory** for context retention
- **Human-in-the-loop** for safety and control

The system is fully functional, uses free/open-source components, and can be extended with additional tools and capabilities. The LangGraph framework proved excellent for orchestrating the complex workflow, and the modular design allows for easy enhancements.

**Key Achievements**:
- ✅ Complete agentic system implementation
- ✅ Multiple free LLM provider options
- ✅ Four functional tools
- ✅ Short-term and long-term memory
- ✅ Human-in-the-loop approval mechanism
- ✅ Working demos and examples

**Future Enhancements**:
- Enhanced tool calling support for local LLMs
- More sophisticated memory retrieval strategies
- Additional tools (database queries, API integrations)
- Interactive approval prompts for HITL
- Multi-agent collaboration capabilities

---

## References

- LangGraph Documentation: https://github.com/langchain-ai/langgraph
- LangChain Documentation: https://python.langchain.com/
- FAISS Documentation: https://github.com/facebookresearch/faiss
- Ollama: https://ollama.ai/
- Google Gemini API: https://ai.google.dev/

---

**Report Length**: ~5 pages  
**Word Count**: ~2,500 words

