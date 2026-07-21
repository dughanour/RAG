from langchain_groq import ChatGroq
from loguru import logger
from app.agents.state import AgentState
from app.config import settings



_llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL, temperature=0.0)

_PLANNER_PROMPT = """You are an intelligent Assistant Planner.
Analyze the conversation history and the latest user message.
CONVERSATION HISTORY:
{history}
LATEST MESSAGE:
"{user_message}"
Task:
1. If the latest message is a greeting (hi, hello) or a question that can be answered 
   using ONLY the conversation history above, respond with exactly: CONVERSATIONAL
2. If it is a technical question that requires documentation retrieval, 
   output a refined search query optimized for semantic search.
Output ONLY 'CONVERSATIONAL' or the search query. Nothing else."""


def planner_node(state: AgentState) -> dict:
    """
    Classifies the user's intent:
      - CONVERSATIONAL → skip retrieval, answer from memory
      - Search query   → route to retriever for document lookup
    """
    
    messages = state.get("messages", [])
    if not messages:
        logger.warning("Planner received empty messages")
        return {
            "current_query": "CONVERSATIONAL",
            "status": "No messages to process",
            "plan": ["Intent: Empty"],
        }
    
    # Build conversation history (all messages except the latest)
    history = ""
    for msg in messages[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = messages[-1]["content"]

    prompt = _PLANNER_PROMPT.format(history=history or "(no prior history)", user_message=user_message)

    logger.info("Planner analyzing query: '{}'", user_message[:80])
    decision = _llm.invoke(prompt).content.strip()
    logger.info("Planner decision: {}", decision)

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"],
        }
    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"],
    }




