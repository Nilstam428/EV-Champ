import React from 'react';
import './ChatMessage.css';

interface ChatMessageProps {
  content: string;
  isUser: boolean;
  timestamp?: string;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ content, isUser, timestamp }) => {
  return (
    <div className={`chat-message ${isUser ? 'user' : 'bot'}`}>
      <div className="message-content">
        <div className="message-text">{content}</div>
        {timestamp && <div className="message-timestamp">{timestamp}</div>}
      </div>
    </div>
  );
};

export default ChatMessage;