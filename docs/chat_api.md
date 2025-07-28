# Chat System API Documentation

## Overview
This document describes the Chat System API, which provides functionality for managing chat conversations, messages, and related features.

## Data Models

### Chat
- **user_id** (string): Unique identifier for the user
- **session_id** (string): Unique identifier for the chat session
- **rag_enabled** (boolean): Flag for RAG (Retrieval-Augmented Generation) feature
- **title** (string, optional): Chat title
- **chat_type** (enum): Type of chat (PERSONAL, GROUP, SYSTEM)
- **status** (enum): Chat status (ACTIVE, ARCHIVED, DELETED)
- **chat_metadata** (object, optional): Additional chat metadata

### Message
- **content** (string): Message content
- **is_user** (boolean): Whether the message is from the user
- **message_type** (enum): Type of message (TEXT, IMAGE, FILE, SYSTEM)
- **status** (enum): Message status (SENDING, SENT, DELIVERED, READ, FAILED)
- **message_metadata** (object, optional): Additional message metadata
- **parent_message_id** (integer, optional): ID of parent message for threading
- **reactions** (object, optional): Message reactions from users

## API Endpoints

### Chat Management

#### Create Chat
```http
POST /api/v1/chats
```
Creates a new chat session.

#### Get Chat
```http
GET /api/v1/chats/{chat_id}
```
Retrieves chat details by ID.

#### List Chats
```http
GET /api/v1/chats
```
Retrieves a list of chats for the user.

#### Update Chat Status
```http
PATCH /api/v1/chats/{chat_id}/status
```
Updates chat status (archive/delete).

### Message Operations

#### Send Message
```http
POST /api/v1/chats/{chat_id}/messages
```
Sends a new message in a chat.

#### Get Messages
```http
GET /api/v1/chats/{chat_id}/messages
```
Retrieves messages from a chat.

#### Update Message Status
```http
PATCH /api/v1/messages/{message_id}/status
```
Updates message status (delivered/read).

### Message Reactions

#### Add Reaction
```http
POST /api/v1/messages/{message_id}/reactions
```
Adds a reaction to a message.

#### Remove Reaction
```http
DELETE /api/v1/messages/{message_id}/reactions/{reaction}
```
Removes a reaction from a message.

## Advanced Features

### Message Threading
Messages can be threaded by setting the `parent_message_id` when creating a new message.

### RAG Support
When `rag_enabled` is true, the chat supports Retrieval-Augmented Generation for enhanced responses.

### Real-time Updates
Message status updates (delivered/read) are handled in real-time.

## Error Handling
The API uses standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 500: Internal Server Error

## Best Practices
1. Always check message status before displaying
2. Handle offline scenarios gracefully
3. Implement proper error handling
4. Use appropriate chat types for different scenarios
5. Manage metadata fields efficiently