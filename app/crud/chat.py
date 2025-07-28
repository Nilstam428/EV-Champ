from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models.chat import Chat, Message, ChatStatus, MessageStatus
from app.schemas.chat import ChatCreate, MessageCreate

def create_chat(db: Session, chat: ChatCreate):
    db_chat = Chat(**chat.dict())
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return db_chat

def get_chat(db: Session, chat_id: int):
    return db.query(Chat).filter(Chat.id == chat_id, Chat.is_deleted == False).first()

def get_chats(db: Session, user_id: str, skip: int = 0, limit: int = 100):
    return db.query(Chat).filter(
        Chat.user_id == user_id,
        Chat.is_deleted == False
    ).order_by(desc(Chat.last_activity)).offset(skip).limit(limit).all()

def update_chat_status(db: Session, chat_id: int, status: ChatStatus):
    db_chat = get_chat(db, chat_id)
    if db_chat:
        db_chat.status = status
        db.commit()
        db.refresh(db_chat)
    return db_chat

def soft_delete_chat(db: Session, chat_id: int):
    return update_chat_status(db, chat_id, ChatStatus.DELETED)

def create_chat_message(db: Session, message: MessageCreate, chat_id: int):
    db_message = Message(**message.dict(), chat_id=chat_id)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    
    # Update chat's last activity
    chat = get_chat(db, chat_id)
    if chat:
        chat.last_activity = datetime.utcnow()
        db.commit()
    
    return db_message

def get_chat_messages(db: Session, chat_id: int, skip: int = 0, limit: int = 100):
    return db.query(Message).filter(
        Message.chat_id == chat_id,
        Message.is_deleted == False
    ).order_by(desc(Message.sent_at)).offset(skip).limit(limit).all()

def update_message_status(db: Session, message_id: int, status: MessageStatus):
    db_message = db.query(Message).filter(Message.id == message_id).first()
    if db_message:
        db_message.status = status
        if status == MessageStatus.DELIVERED:
            db_message.delivered_at = datetime.utcnow()
        elif status == MessageStatus.READ:
            db_message.read_at = datetime.utcnow()
        db.commit()
        db.refresh(db_message)
    return db_message

def add_message_reaction(db: Session, message_id: int, reaction: str, user_id: str):
    db_message = db.query(Message).filter(Message.id == message_id).first()
    if db_message:
        if not db_message.reactions:
            db_message.reactions = {}
        if reaction not in db_message.reactions:
            db_message.reactions[reaction] = []
        if user_id not in db_message.reactions[reaction]:
            db_message.reactions[reaction].append(user_id)
            db.commit()
            db.refresh(db_message)
    return db_message

def remove_message_reaction(db: Session, message_id: int, reaction: str, user_id: str):
    db_message = db.query(Message).filter(Message.id == message_id).first()
    if db_message and db_message.reactions and reaction in db_message.reactions:
        if user_id in db_message.reactions[reaction]:
            db_message.reactions[reaction].remove(user_id)
            if not db_message.reactions[reaction]:
                del db_message.reactions[reaction]
            db.commit()
            db.refresh(db_message)
    return db_message

def get_or_create_chat(db: Session, user_id: str, session_id: str, rag_enabled: bool = False):
    chat = db.query(Chat).filter(Chat.user_id == user_id, Chat.session_id == session_id, Chat.is_deleted == False).first()
    if chat:
        return chat
    else:
        new_chat = ChatCreate(user_id=user_id, session_id=session_id, rag_enabled=rag_enabled)
        return create_chat(db=db, chat=new_chat)

def get_chat_by_session(db: Session, user_id: str, session_id: str):
    return db.query(Chat).filter(Chat.user_id == user_id, Chat.session_id == session_id, Chat.is_deleted == False).first()
