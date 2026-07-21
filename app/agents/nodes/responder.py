from langchain_groq import ChatGroq
from loguru import logger
from app.agents.state import AgentState
from app.config import settings

_llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL, temperature=0.0)

_MAX_CONTEXT_CHARS = 25_000

_CONVERSATIONAL_PROMPT = """You are a friendly and helpful Enterprise AI Assistant.
Answer the user's latest message using the CONVERSATION HISTORY below.
CONVERSATION HISTORY:
{history}
LATEST MESSAGE:
"{user_message}"
"""
_TECHNICAL_PROMPT = """You are a Senior Technical Architect.
Answer the question using the TECHNICAL CONTEXT provided.
If the context does not contain the answer, say so honestly.
TECHNICAL CONTEXT:
{context}
CONVERSATION HISTORY:
{history}
USER QUESTION:
"{user_message}"
"""

def responder_node(state: AgentState) -> dict:
    """
    Synthesizes a final response using either:
      - Conversation history only (CONVERSATIONAL)
      - Retrieved documents + history (TECHNICAL)
    """
    query = state["current_query"]
    messages = state.get("messages", [])

    # Build conversation history
    history = ""
    for msg in messages[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = messages[-1]["content"] if messages else ""

    if query == "CONVERSATIONAL":
        logger.info("Generating conversational response from memory")
        prompt = _CONVERSATIONAL_PROMPT.format(
            history=history or "(no prior history)",
            user_message=user_message,
        )

    else:
        logger.info("Generating technical RAG response")
        # Truncate context to fit model token limits
        context = ""
        for doc in state.get("documents", []):
            if len(context) + len(doc) < _MAX_CONTEXT_CHARS:
                context += doc + "\n\n"
            else:
                logger.warning("Context truncated to fit token limits")
                break
        prompt = _TECHNICAL_PROMPT.format(
            context=context or "(no context available)",
            history=history or "(no prior history)",
            user_message=user_message,
        )
    
    try:
        content = _llm.invoke(prompt).content.strip()
        logger.info("Response generated successfully")
        return {
            "final_answer": content,
            "status": "Response generated",
            "plan": state["plan"],
            "messages": [{"role": "assistant", "content": content}],
        }
    except Exception as e:
        logger.exception("Failed to generate response: {}", e)
        raise



  

        
    