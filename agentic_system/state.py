"""
State management for the agentic system using LangGraph StateGraph
"""
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State structure for the agentic system"""
    messages: List[Dict[str, Any]]  # Conversation history (short-term memory)
    current_query: str  # Current user query
    tool_results: List[Dict[str, Any]]  # Results from tool executions
    requires_approval: bool  # Flag for human-in-the-loop
    approval_pending: Optional[str]  # Action waiting for approval
    memory_context: List[str]  # Context retrieved from long-term memory
    step_count: int  # Track number of steps taken

