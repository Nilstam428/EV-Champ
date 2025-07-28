from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from app.models.chat import ChatType, ChatStatus, MessageType, MessageStatus

class ChatBase(BaseModel):
    user_id: str
    session_id: str
    rag_enabled: bool = False
    title: Optional[str] = None
    chat_type: ChatType = ChatType.PERSONAL
    status: ChatStatus = ChatStatus.ACTIVE
    metadata: Optional[Dict[str, Any]] = None

class ChatCreate(ChatBase):
    pass

class ChatResponse(ChatBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime]
    message_count: int = 0

    class Config:
        from_attributes = True

class MessageBase(BaseModel):
    content: str
    is_user: bool = True
    message_type: MessageType = MessageType.TEXT
    status: MessageStatus = MessageStatus.SENDING
    message_metadata: Optional[Dict[str, Any]] = None
    parent_message_id: Optional[int] = None
    reactions: Optional[Dict[str, Any]] = None

class MessageCreate(MessageBase):
    pass

class MessageResponse(MessageBase):
    id: int
    chat_id: int
    created_at: datetime
    updated_at: Optional[datetime]
    sent_at: datetime
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class ChatMessageRequest(BaseModel):
    content: str
    user_id: str
    session_id: str
    rag_enabled: bool = False

class ChatMessageResponse(BaseModel):
    user_message: MessageResponse
    bot_message: MessageResponse
    chat: ChatResponse