"""
Interactive demo for the Agentic AI System
Shows detailed step-by-step execution
"""
from typing import Any
from agentic_system import AgenticSystem
import json


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step_num: int, description: str, data: Any = None):
    """Print a formatted step"""
    print(f"\n[Step {step_num}] {description}")
    if data:
        if isinstance(data, dict):
            print(json.dumps(data, indent=2))
        else:
            print(str(data))


def main():
    """Run interactive demonstration"""
    print_section("SWE645 - Agentic AI System - Interactive Demo")
    
    try:
        # Initialize
        print_step(1, "Initializing Agentic System")
        
        # Try Ollama first (no quotas), fallback to Gemini
        try:
            print("Trying Ollama (no quotas)...")
            agent = AgenticSystem(provider="ollama", model_name="llama3.2")
            print("[OK] System initialized with Ollama")
            print(f"  - LLM: Ollama (llama3.2) - Unlimited requests!")
        except Exception as ollama_error:
            print(f"[WARNING] Ollama not available: {ollama_error}")
            print("   Falling back to Google Gemini...")
            agent = AgenticSystem(provider="gemini")
            print("[OK] System initialized with Gemini")
            print("  - LLM: Google Gemini (free tier - 5 req/min)")
        
        print(f"  - Tools: {len(agent.tools)} available")
        print(f"  - Memory: Short-term (conversation) + Long-term (FAISS)")
        
        # Demo queries
        queries = [
            "Calculate 42 * 17",
            "What is the square root of 144?",
            "Search for information about LangGraph framework"
        ]
        
        for i, query in enumerate(queries, 1):
            print_section(f"Demo Query {i}")
            print(f"User Query: {query}\n")
            
            # Run with detailed output
            result = agent.run_interactive(query)
            
            print_step(2, "Execution Results")
            print(f"  Steps taken: {result['steps']}")
            print(f"  Tools used: {result['tools_used']}")
            print(f"  Memory context retrieved: {result['memory_used']}")
            
            print_step(3, "Agent Response")
            print(result['response'])
        
        # Show architecture
        print_section("System Architecture")
        print("""
        ┌─────────────────────────────────────────────────────────┐
        │              Agentic AI System (LangGraph)              │
        ├─────────────────────────────────────────────────────────┤
        │                                                           │
        │  1. Retrieve Memory (Long-term)                          │
        │     ↓                                                     │
        │  2. Agent Node (LLM Reasoning)                          │
        │     ↓                                                     │
        │  3. Tools Node (Execute Actions)                         │
        │     ↓                                                     │
        │  4. Check Approval (Human-in-the-Loop)                   │
        │     ↓                                                     │
        │  5. Respond (Generate Final Answer)                       │
        │     ↓                                                     │
        │  6. Store Memory (Save to Long-term)                     │
        │                                                           │
        └─────────────────────────────────────────────────────────┘
        
        Components:
        • LLM: Ollama (Llama 3.2) or Google Gemini (Reasoning & Planning)
        • Tools: Web Search, Calculator, File Operations
        • Short-term Memory: Conversation History
        • Long-term Memory: FAISS Vector Store
        • Human-in-the-Loop: Approval for Critical Actions
        """)
        
        print_section("Demo Complete")
        print("The agentic system successfully demonstrated:")
        print("  [OK] LLM-based reasoning and planning")
        print("  [OK] Tool selection and execution")
        print("  [OK] Short-term memory (conversation context)")
        print("  [OK] Long-term memory (vector store)")
        print("  [OK] Human-in-the-loop approval mechanism")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nTroubleshooting:")
        print("1. For Ollama (FREE, recommended):")
        print("   - Install: https://ollama.ai")
        print("   - Run: ollama pull llama3.2")
        print("   - No API key needed!")
        print("\n2. For Google Gemini (FREE):")
        print("   - Get free API key: https://aistudio.google.com/app/apikey")
        print("   - Add to .env: GOOGLE_API_KEY=your_key")
        print("   - Free tier: 5 req/min for gemini-2.5-flash")
        print("\n3. For Hugging Face (FREE with token):")
        print("   - Get free token: https://huggingface.co/settings/tokens")
        print("   - Add to .env: HUGGINGFACE_API_KEY=your_token")
        print("\n4. Install dependencies: pip install -r requirements.txt")


if __name__ == "__main__":
    main()

