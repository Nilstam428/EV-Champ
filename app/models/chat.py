from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, JSON, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.db.database import Base
from app.schemas.base import BaseModel
#  from sqlalchemy.orm import remote

class ChatType(enum.Enum):
    PERSONAL = "personal"
    GROUP = "group"
    SYSTEM = "system"

class ChatStatus(enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

class MessageType(enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    SYSTEM = "system"

class MessageStatus(enum.Enum):
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"

class Chat(Base, BaseModel):
    __tablename__ = "chats"
    
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, index=True, nullable=False)
    rag_enabled = Column(Boolean, default=False)
    title = Column(String, nullable=True)
    chat_type = Column(Enum(ChatType), default=ChatType.PERSONAL, nullable=False)
    status = Column(Enum(ChatStatus), default=ChatStatus.ACTIVE, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    chat_metadata = Column(JSON, nullable=True)
    is_deleted = Column(Boolean, default=False, index=True)
    
    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")

class Message(Base, BaseModel):
    __tablename__ = "messages"
    
    chat_id = Column(Integer, ForeignKey("chats.id"), nullable=False)
    content = Column(Text, nullable=False)
    is_user = Column(Boolean, default=True)
    message_type = Column(Enum(MessageType), default=MessageType.TEXT, nullable=False)
    status = Column(Enum(MessageStatus), default=MessageStatus.SENDING, nullable=False)
    sent_at = Column(DateTime, default=datetime.utcnow)
    delivered_at = Column(DateTime, nullable=True)
    read_at = Column(DateTime, nullable=True)
    message_metadata = Column(JSON, nullable=True)
    parent_message_id = Column(Integer, ForeignKey("messages.id"), nullable=True)
    reactions = Column(JSON, nullable=True)
    is_deleted = Column(Boolean, default=False, index=True)
    chat = relationship("Chat", back_populates="messages")
    # parent_message = relationship("Message", remote_side=[remote(Message.id)], backref="replies", foreign_keys=[parent_message_id])
