# Chatbot API

This project is a FastAPI-based chatbot.

## Project Structure

```
app/
├── crud/         # Contains CRUD operations for database models.
├── models/       # Contains SQLAlchemy database models.
├── routers/      # Contains API routes.
│   └── v1/       # API version 1.
├── schemas/      # Contains Pydantic schemas.
├── services/     # Contains llm logic and services.
├── core/         # Contains core application settings and configurations.
├── db/           # Contains database connection and session management.
└── main.py       # Main application file.
```

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure environment variables:**

    Create a `.env` file in the project root and add the necessary variables.

3.  **Run the application:**
    ```bash
    uvicorn app.main:app --reload
    ```


## Frontend Setup and Run

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Run the frontend application:**
    ```bash
    npm run dev
    ```

The frontend application will typically be available at `http://localhost:5173/` or a similar address, as indicated in your terminal.

## API Documentation

Access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/api/v1/docs` after the backend server is running.

## API Endpoints
`http://localhost:5173/`
### Chat

#### Create a new chat

`POST /api/v1/chat/`

**Request Body:**

```json
{
  "user_id": "string",
  "rag_enabled": false
}
```

**Response:**

```json
{
  "user_id": "string",
  "rag_enabled": false,
  "id": 0,
  "created_at": "2023-10-27T10:00:00.000Z",
  "updated_at": "2023-10-27T10:00:00.000Z"
}
```

#### Get a chat by ID

`GET /api/v1/chat/{chat_id}`

**Response:**

```json
{
  "user_id": "string",
  "rag_enabled": false,
  "id": 0,
  "created_at": "2023-10-27T10:00:00.000Z",
  "updated_at": "2023-10-27T10:00:00.000Z"
}
```

#### Get all chats

`GET /api/v1/chat/`

**Response:**

```json
[
  {
    "user_id": "string",
    "rag_enabled": false,
    "id": 0,
    "created_at": "2023-10-27T10:00:00.000Z",
    "updated_at": "2023-10-27T10:00:00.000Z"
  }
]
```

#### Create a message for a chat

`POST /api/v1/chat/{chat_id}/messages/`

**Request Body:**

```json
{
  "content": "string",
  "is_user": true
}
```

**Response:**

```json
{
  "content": "string",
  "is_user": true,
  "id": 0,
  "chat_id": 0,
  "created_at": "2023-10-27T10:00:00.000Z",
  "updated_at": "2023-10-27T10:00:00.000Z"
}
```

#### Get messages for a chat

`GET /api/v1/chat/{chat_id}/messages/`

**Response:**

```json
[
  {
    "content": "string",
    "is_user": true,
    "id": 0,
    "chat_id": 0,
    "created_at": "2023-10-27T10:00:00.000Z",
    "updated_at": "2023-10-27T10:00:00.000Z"
  }
]
```