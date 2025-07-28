from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.database import Base
from app.schemas.base import BaseModel

class User(Base, BaseModel):
    __tablename__ = "users"
    
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="user")
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")