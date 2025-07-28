import React, { useState, useEffect, useRef } from 'react';
import ChatMessage from './ChatMessage';
import './ChatContainer.css';

interface Message {
  content: string;
  isUser: boolean;
  timestamp: string;
  id?: number;
}

const ChatContainer: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [chatId, setChatId] = useState<number | null>(null);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [ragEnabled, setRagEnabled] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const generateSessionId = () => {
    return 'session-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
  };

  const createNewChat = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/chat/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: 'default-user',
          session_id: generateSessionId(),
          rag_enabled: false
        })
      });
      const data = await response.json();
      setChatId(data.id);
    } catch (error) {
      console.error('Error creating chat:', error);
    }
  };

  useEffect(() => {
    createNewChat();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !chatId) return;

    const messageContent = input.trim();
    setIsLoading(true);

    const userMessage: Message = {
      content: messageContent,
      isUser: true,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');

    try {
      const response = await fetch(`http://localhost:8000/api/v1/chat/${chatId}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: messageContent,
          is_user: true,
          message_type: 'text',
          status: 'sending',
          message_metadata: null,
          parent_message_id: null,
          reactions: null
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const botMessage: Message = {
        content: data.bot_message?.content || data.content || data.message?.content || 'No content',
        isUser: false,
        timestamp: data.bot_message?.created_at || data.created_at || data.message?.created_at || new Date().toISOString(),
        id: data.bot_message?.id || data.id || data.message?.id
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, {
        content: 'Sorry, there was an error processing your message.',
        isUser: false,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsLoading(false);
    }
};

  const handleClearChat = () => {
    setMessages([]);
  };

  const handleExportChat = () => {
    const chatContent = messages
      .map(msg => `${msg.isUser ? 'User' : 'Bot'}: ${msg.content}\n${msg.timestamp}`)
      .join('\n\n');
    const blob = new Blob([chatContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${new Date().toISOString()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  const toggleSettings = () => {
    setIsSettingsOpen(!isSettingsOpen);
  };

  return (
    <div className={`chat-container ${isDarkMode ? 'dark-mode' : ''}`}>
      <div className="header-container">
        <div className="header-options">
          <button onClick={handleClearChat} className="option-button">
            Clear Chat
          </button>
          <button onClick={handleExportChat} className="option-button">
            Export Chat
          </button>
          <button onClick={toggleTheme} className="option-button">
            {isDarkMode ? '🌞 Light' : '🌙 Dark'}
          </button>
          <button onClick={toggleSettings} className="option-button">
            ⚙️ Settings
          </button>
        </div>
      </div>
      {isSettingsOpen && (
        <div className="settings-panel">
          <h3>Settings</h3>
          <div className="setting-option">
            <label>
              <input
                type="checkbox"
                checked={ragEnabled}
                onChange={(e) => setRagEnabled(e.target.checked)}
              />
              Enable RAG
            </label>
          </div>
        </div>
      )}
      <div className="messages-container">
        {messages.map((message, index) => (
          <ChatMessage
            key={index}
            content={message.content}
            isUser={message.isUser}
            timestamp={message.timestamp}
          />
        ))}
        {isLoading && (
          <div className="loading-message">
            <div className="loading-dots">
              <div></div>
              <div></div>
              <div></div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <form className="input-container" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatContainer;