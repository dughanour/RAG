from typing import TypedDict, Annotated
import operator
class AgentState(TypedDict):
    """Shared state maintained across all nodes in the agentic RAG pipeline."""
    messages: Annotated[list[dict], operator.add]  # Chat history (auto-appended by LangGraph)
    current_query: str                              # Planner output: "CONVERSATIONAL" or refined search query
    documents: list[str]                            # Retrieved chunks from vector DB
    plan: list[str]                                 # Trace of decisions made by each node
    status: str                                     # Human-readable status of current node
    final_answer: str                               # LLM-generated response to the user