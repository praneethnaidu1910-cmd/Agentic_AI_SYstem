"""
Main entry point for the Agentic AI System
Demonstrates basic usage
"""
from agentic_system import AgenticSystem


def main():
    """Run basic demonstration"""
    print("=" * 60)
    print("SWE645 - Agentic AI System Demo")
    print("=" * 60)
    print()
    
    try:
        # Initialize the agent
        # OPTION 1: Ollama (RECOMMENDED - No quotas, 100% free, local)
        #   1. Install: https://ollama.ai
        #   2. Run: ollama pull llama3.2 (or llama3.1, mistral, mixtral)
        #   3. Use: provider="ollama"
        
        # OPTION 2: Google Gemini (Free but has quotas - 5 req/min for gemini-2.5-flash)
        #   Use: provider="gemini"
        
        # OPTION 3: Hugging Face (Free with token)
        #   Use: provider="huggingface"
        
        print("Initializing agentic system...")
        print("TIP: If you hit quota limits, switch to Ollama (no quotas!)")
        print("   Change provider to 'ollama' in this file\n")
        
        # Try Ollama first (no quotas), fallback to Gemini
        try:
            print("Trying Ollama (no quotas)...")
            # Use llama3.2 for better function calling support (you already have it!)
            # Other options: "llama3.1", "mistral", "mixtral"
            agent = AgenticSystem(provider="ollama", model_name="llama3.2")
            print("[OK] Using Ollama - Unlimited requests!\n")
        except Exception as ollama_error:
            print(f"[WARNING] Ollama not available: {ollama_error}")
            print("   Falling back to Google Gemini...")
            print("   (Note: Gemini free tier has low quotas - 5 req/min)\n")
            agent = AgenticSystem(provider="gemini")
        
        print("[OK] Agent initialized successfully\n")
        
        # Example 1: Simple calculation
        print("Example 1: Calculator Tool")
        print("-" * 60)
        query1 = "Calculate 15 * 23"
        print(f"Query: {query1}")
        response1 = agent.run(query1)
        print(f"Response: {response1}\n")
        
        # Example 2: Web search (mock)
        print("Example 2: Web Search Tool")
        print("-" * 60)
        query2 = "What is the capital of France?"
        print(f"Query: {query2}")
        response2 = agent.run(query2)
        print(f"Response: {response2}\n")
        
        # Example 3: Combined query
        print("Example 3: Multi-tool Query")
        print("-" * 60)
        query3 = "What is 25 squared? Then search for information about AI agents."
        print(f"Query: {query3}")
        response3 = agent.run(query3)
        print(f"Response: {response3}\n")
        
        print("=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nFREE Options Setup:")
        print("1. For Google Gemini (FREE - Recommended):")
        print("   - Get free API key: https://aistudio.google.com/app/apikey")
        print("   - Add to .env: GOOGLE_API_KEY=your_key")
        print("   - Free tier: 60 req/min, 1,500 req/day")
        print("\n2. For Hugging Face (FREE with token):")
        print("   - Get free token: https://huggingface.co/settings/tokens")
        print("   - Add to .env: HUGGINGFACE_API_KEY=your_token")
        print("   - Use: AgenticSystem(provider='huggingface')")
        print("\n3. For Ollama (FREE, local, no API key):")
        print("   - Install: https://ollama.ai")
        print("   - Run: ollama pull llama3.2 (better function calling)")
        print("   - Use: AgenticSystem(provider='ollama', model_name='llama3.2')")
        print("\n4. Install dependencies: pip install -r requirements.txt")


if __name__ == "__main__":
    main()

