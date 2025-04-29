# Multi-LLM Chat & Image API

## Description

This is a FastAPI backend application that serves as a unified interface for interacting with multiple Large Language Models (LLMs) and an image generation service. It currently supports:

*   **Chat:**
    *   OpenAI 
    *   Google Gemini 
    *   xAI Grok 
*   **Image Generation:**
    *   OpenAI (DALL-E 3)

The API allows clients (like a frontend application) to specify the desired provider and send messages, maintaining conversation context via a history parameter managed by the client.

## Prerequisites

*   Python 3.8+
*   pip (Python package installer)
*   API Keys:
    *   OpenAI API Key
    *   Google AI API Key (for Gemini)
    *   xAI API Key (for Grok)

## Setup

1.  **Clone or Download:** Get the project files.
    ```bash
    # If using Git
    # git clone <your-repository-url>
    # cd <repository-name>
    ```

2.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate Virtual Environment:**
    *   macOS/Linux: `source venv/bin/activate`
    *   Windows: `.\venv\Scripts\activate`

4.  **Install Dependencies:**
    ```bash
    pip install fastapi uvicorn python-dotenv openai google-generativeai httpx
    ```
    *(Optional: Create a `requirements.txt` file with `pip freeze > requirements.txt` for easier dependency management)*

5.  **Create `.env` File:**
    Create a file named `.env` in the project root directory and add your API keys:
    ```.env
    OPENAI_API_KEY=your_openai_api_key_here
    GOOGLE_API_KEY=your_google_gemini_api_key_here
    XAI_API_KEY=your_xai_grok_api_key_here
    ```
    **Important:** Add `.env` to your `.gitignore` file if using Git.

6.  **Run the Server:**
    ```bash
    uvicorn main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

### 1. Root

*   **URL:** `/`
*   **Method:** `GET`
*   **Description:** Returns a welcome message and lists available services based on configured API keys.
*   **Success Response (200 OK):**
    ```json
    {
      "message": "Welcome to the Multi-LLM API!",
      "available_services": [
        "OpenAI Chat",
        "OpenAI Image",
        "Gemini Chat",
        "xAI Grok Chat"
      ],
      "docs_url": "/docs"
    }
    ```

### 2. Chat

*   **URL:** `/chat`
*   **Method:** `POST`
*   **Description:** Sends a message to the specified LLM provider, including conversation history for context.
*   **Request Body:**
    ```json
    {
      "model_provider": "string (openai | gemini | xai)",
      "message": "string",
      "history": [
        {
          "role": "string (user | assistant | system)",
          "content": "string"
        }
      ],
      "model_name": "string (optional)"
    }
    ```
    *   `model_provider`: Required. Selects the backend service.
    *   `message`: Required. The latest message from the user.
    *   `history`: Optional (but needed for follow-up turns). List of previous messages in the current session. The client must manage and send this list.
    *   `model_name`: Optional. Allows specifying a non-default model for the selected provider (e.g., "gpt-3.5-turbo" for OpenAI).

*   **Success Response (200 OK):**
    ```json
    {
      "response": "string (The LLM's reply)",
      "model_used": "string (Identifier of the model that responded)"
    }
    ```
*   **Example Request (OpenAI, second turn):**
    ```json
    {
      "model_provider": "openai",
      "message": "Tell me more about that.",
      "history": [
        {
          "role": "user",
          "content": "What is FastAPI?"
        },
        {
          "role": "assistant",
          "content": "FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints."
        }
      ]
    }
    ```

### 3. Generate Image

*   **URL:** `/generate-image`
*   **Method:** `POST`
*   **Description:** Generates an image using OpenAI's DALL-E 3 based on a text prompt.
*   **Request Body:**
    ```json
    {
      "prompt": "string"
    }
    ```
*   **Success Response (200 OK):**
    ```json
    {
      "image_url": "string (URL of the generated image)",
      "model_used": "dall-e-3"
    }
    ```
*   **Example Request:**
    ```json
    {
      "prompt": "An astronaut riding a bicycle on the moon, digital art"
    }
    ```

### API Documentation

Interactive API documentation (Swagger UI) is available at `/docs` when the server is running (e.g., `http://127.0.0.1:8000/docs`).