from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retriever_node
from app.agents.nodes.responder import responder_node


def _route_after_planner(state: AgentState) -> str:
    """Route based on planner's classification."""
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"
    return "retriever"

# Build the graph
workflow = StateGraph(AgentState)
# Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("responder", responder_node)
# Define edges
workflow.set_entry_point("planner")
workflow.add_conditional_edges(
    "planner",
    _route_after_planner,
    {
        "retriever": "retriever",
        "responder": "responder",
    },
)
workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)
# Compile with memory checkpointing (enables multi-turn conversation)
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)