"""
Tools for the agentic system: Web search, calculator, file operations
"""
import os
import json
import requests
from typing import Dict, Any, Optional
from langchain.tools import tool
from langchain_core.tools import ToolException


@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for information. Use this when you need current information
    or facts that might not be in your training data.
    
    Args:
        query: The search query string
        
    Returns:
        Search results as a string
    """
    try:
        # Try using SerpAPI if available
        api_key = os.getenv("SERPAPI_API_KEY")
        if api_key:
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": api_key,
                "engine": "google"
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = []
                if "organic_results" in data:
                    for result in data["organic_results"][:3]:
                        results.append(f"Title: {result.get('title', 'N/A')}\nSnippet: {result.get('snippet', 'N/A')}")
                return "\n\n".join(results) if results else "No results found."
        
        # Fallback: Mock search results for demonstration
        return f"[MOCK SEARCH] Results for '{query}': Found relevant information. (Note: Add SERPAPI_API_KEY to .env for real web search)"
    
    except Exception as e:
        return f"Error performing web search: {str(e)}"


@tool
def calculator_tool(expression: str) -> str:
    """
    Perform mathematical calculations. Use this for any arithmetic operations.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "15 * 23", "sqrt(16)")
        
    Returns:
        The calculation result
    """
    try:
        # Safe evaluation of mathematical expressions
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "__builtins__": {}
        }
        
        # Add math functions
        import math
        allowed_names.update({
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "exp": math.exp,
            "pi": math.pi, "e": math.e
        })
        
        result = eval(expression, allowed_names)
        return f"Result: {result}"
    
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def file_read_tool(file_path: str) -> str:
    """
    Read the contents of a file. Use this to retrieve information from files.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        File contents as a string
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."
        
        if not os.path.isfile(file_path):
            return f"Error: '{file_path}' is not a file."
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return f"File contents of '{file_path}':\n\n{content}"
    
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def file_write_tool(file_path: str, content: str, requires_approval: bool = True) -> str:
    """
    Write content to a file. This action requires human approval if requires_approval is True.
    Use this to save information or create files.
    
    Args:
        file_path: Path where the file should be written
        content: Content to write to the file
        requires_approval: Whether this action needs human approval (default: True)
        
    Returns:
        Status message indicating if the write was successful or pending approval
    """
    if requires_approval:
        return f"APPROVAL_REQUIRED: Write to '{file_path}'. Content preview: {content[:100]}..."
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully wrote to '{file_path}'"
    
    except Exception as e:
        return f"Error writing file: {str(e)}"


def get_tools():
    """Return all available tools"""
    return [
        web_search_tool,
        calculator_tool,
        file_read_tool,
        file_write_tool
    ]

