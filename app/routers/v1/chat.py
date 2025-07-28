from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.db.database import get_db
from app.schemas.chat import (
    ChatCreate, ChatResponse, MessageCreate, MessageResponse,
    ChatMessageRequest, ChatMessageResponse
)
from app.crud import chat as crud_chat
from app.services.llm_service import LLMService

router = APIRouter()

# Global LLM service instance with lazy initialization
_llm_service = None

def get_llm_service():
    """Get or create LLM service instance with proper error handling"""
    global _llm_service
    if _llm_service is None:
        try:
            _llm_service = LLMService()
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize LLM service: {str(e)}"
            )
    return _llm_service

@router.post("/message", response_model=ChatMessageResponse)
async def send_message(request: ChatMessageRequest, db: Session = Depends(get_db)):
    """Send a message and get bot response"""
    try:
        # Get or create chat session
        chat = crud_chat.get_or_create_chat(
            db=db,
            user_id=request.user_id,
            session_id=request.session_id,
            rag_enabled=request.rag_enabled
        )
        
        # Create user message
        user_message_data = MessageCreate(content=request.content, is_user=True)
        user_message = crud_chat.create_chat_message(
            db=db, message=user_message_data, chat_id=chat.id
        )
        
        # Get chat history for context
        chat_history = crud_chat.get_chat_messages(
            db=db, chat_id=chat.id
        )
        
        # Generate bot response with proper error handling
        try:
            llm_service = get_llm_service()
            bot_response = await llm_service.generate_response(
                query=request.content,
                chat_history=chat_history,
                rag_enabled=request.rag_enabled
            )
        except Exception as llm_error:
            # If LLM service fails, provide a fallback response
            bot_response = f"I apologize, but I'm currently experiencing technical difficulties. Please try again later. (Error: {str(llm_error)[:100]})"
        
        # Create bot message
        bot_message_data = MessageCreate(content=bot_response, is_user=False)
        bot_message = crud_chat.create_chat_message(
            db=db, message=bot_message_data, chat_id=chat.id
        )
        
        # Get updated chat with message count
        db.refresh(chat)
        chat_response = ChatResponse(
            id=chat.id,
            user_id=chat.user_id,
            session_id=chat.session_id,
            rag_enabled=chat.rag_enabled,
            title=chat.title,
            created_at=chat.created_at,
            updated_at=chat.updated_at,
            message_count=len(chat.messages)
        )
        
        return ChatMessageResponse(
            user_message=user_message,
            bot_message=bot_message,
            chat=chat_response
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.post("/", response_model=ChatResponse)
def create_chat(chat_in: ChatCreate, db: Session = Depends(get_db)):
    """Create a new chat session"""
    return crud_chat.create_chat(db=db, chat=chat_in)

@router.get("/user/{user_id}", response_model=List[ChatResponse])
def get_user_chats(user_id: str, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all chats for a user"""
    chats = crud_chat.get_chats(db, user_id=user_id, skip=skip, limit=limit)
    return [
        ChatResponse(
            id=chat.id,
            user_id=chat.user_id,
            session_id=chat.session_id,
            rag_enabled=chat.rag_enabled,
            title=chat.title,
            created_at=chat.created_at,
            updated_at=chat.updated_at,
            message_count=len(chat.messages)
        )
        for chat in chats
    ]

@router.get("/{chat_id}", response_model=ChatResponse)
def read_chat(chat_id: int, db: Session = Depends(get_db)):
    """Get specific chat by ID"""
    db_chat = crud_chat.get_chat(db, chat_id=chat_id)
    if db_chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    return ChatResponse(
        id=db_chat.id,
        user_id=db_chat.user_id,
        session_id=db_chat.session_id,
        rag_enabled=db_chat.rag_enabled,
        title=db_chat.title,
        created_at=db_chat.created_at,
        updated_at=db_chat.updated_at,
        message_count=len(db_chat.messages)
    )

@router.get("/{chat_id}/messages", response_model=List[MessageResponse])
def read_chat_messages(
    chat_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get messages for a specific chat"""
    db_chat = crud_chat.get_chat(db, chat_id=chat_id)
    if db_chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    messages = crud_chat.get_chat_messages(db, chat_id=chat_id, skip=skip, limit=limit)
    return messages

@router.post("/{chat_id}/messages", response_model=ChatMessageResponse)
async def create_chat_message(chat_id: int, message: MessageCreate, db: Session = Depends(get_db)):
    """Create a new message in a chat and generate bot response"""
    try:
        db_chat = crud_chat.get_chat(db, chat_id=chat_id)
        if db_chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Create user message
        user_message = crud_chat.create_chat_message(db=db, message=message, chat_id=chat_id)
        
        # Get chat history for context
        chat_history = crud_chat.get_chat_messages(db=db, chat_id=chat_id)
        
        # Generate bot response with proper error handling
        try:
            llm_service = get_llm_service()
            bot_response = await llm_service.generate_response(
                query=message.content,
                chat_history=chat_history,
                rag_enabled=db_chat.rag_enabled
            )
        except Exception as llm_error:
            # If LLM service fails, provide a fallback response
            bot_response = f"I apologize, but I'm currently experiencing technical difficulties. Please try again later. (Error: {str(llm_error)[:100]})"
        
        # Create bot message
        bot_message_data = MessageCreate(content=bot_response, is_user=False)
        bot_message = crud_chat.create_chat_message(db=db, message=bot_message_data, chat_id=chat_id)
        
        # Get updated chat
        db.refresh(db_chat)
        chat_response = ChatResponse(
            id=db_chat.id,
            user_id=db_chat.user_id,
            session_id=db_chat.session_id,
            rag_enabled=db_chat.rag_enabled,
            title=db_chat.title,
            created_at=db_chat.created_at,
            updated_at=db_chat.updated_at,
            message_count=len(db_chat.messages)
        )
        
        return ChatMessageResponse(
            user_message=user_message,
            bot_message=bot_message,
            chat=chat_response
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.get("/session/{user_id}/{session_id}/messages", response_model=List[MessageResponse])
def get_session_messages(
    user_id: str,
    session_id: str,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get messages for a specific user session"""
    chat = crud_chat.get_chat_by_session(db=db, user_id=user_id, session_id=session_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = crud_chat.get_chat_messages(db, chat_id=chat.id, skip=skip, limit=limit)
    return messages

# Health check endpoint for LLM service
@router.get("/health/llm")
async def check_llm_health():
    """Check if LLM service is working properly"""
    try:
        llm_service = get_llm_service()
        # Try a simple test query
        test_response = await llm_service.generate_response(
            query="Hello",
            chat_history=[],
            rag_enabled=False
        )
        return {"status": "healthy", "message": "LLM service is working"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}