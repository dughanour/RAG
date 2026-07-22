import uuid
from loguru import logger
from fastapi import APIRouter, HTTPException
from app.api.schemas import ChatRequest, ChatResponse, NewSessionResponse
from app.agents.graph import app as agent_app

router = APIRouter()

@router.post("/chat/new", response_model=NewSessionResponse)
def new_session():
    """Generate a new unique session ID"""
    session_id = str(uuid.uuid4())
    return NewSessionResponse(session_id=session_id)

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send a message through the Agentic RAG pipeline."""
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        inputs = {"messages": [{"role": "user", "content": request.message}]}

        result = agent_app.invoke(inputs, config=config)

        return ChatResponse(
            answer=result.get("final_answer", ""),
            session_id=request.session_id,
            planner_decision=result.get("current_query", ""),
            status=result.get("status", ""),
            plan=result.get("plan", []),
            retrieved_chunks=len(result.get("documents", [])),
        )
    except Exception as e:
        logger.exception("Chat endpoint failed: {}", e)
        raise HTTPException(status_code=500, detail="Failed to process your message. Please try again.")



