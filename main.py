import os
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from enum import Enum
from typing import List, Optional, Dict, Any

# Import specific API libraries and error types
import openai
import google.generativeai as genai
from google.generativeai.types import generation_types as gemini_types
from openai import APIError as OpenAI_APIError # Alias to avoid name clash if needed

# --- Configuration ---
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY") # For xAI Grok

DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o"
DEFAULT_OPENAI_IMAGE_MODEL = "dall-e-3"
DEFAULT_GEMINI_MODEL = 'gemini-2.0-flash'
DEFAULT_XAI_MODEL = "grok-3-beta"
XAI_BASE_URL = "https://api.x.ai/v1"

# --- API Client Initialization ---
openai_client = None
gemini_model = None
xai_client = None

# Initialize OpenAI Client
if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        # Optional: Add a light check like listing models if needed
        print(f"OpenAI client initialized. Default chat model: {DEFAULT_OPENAI_CHAT_MODEL}, Image: {DEFAULT_OPENAI_IMAGE_MODEL}")
    except Exception as e:
        print(f"Warning: Failed to initialize OpenAI client: {e}")
        openai_client = None
else:
    print("Warning: OPENAI_API_KEY not found in .env. OpenAI features disabled.")

# Initialize Google Gemini Client
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        print(f"Google Gemini configured. Default model: {DEFAULT_GEMINI_MODEL}")
        gemini_model = genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
    except Exception as e:
        print(f"Warning: Failed to configure Google Gemini: {e}")
        gemini_model = None
else:
    print("Warning: GOOGLE_API_KEY not found in .env. Gemini features disabled.")

if XAI_API_KEY:
    try:
        xai_client = openai.OpenAI(
            api_key=XAI_API_KEY,
            base_url=XAI_BASE_URL,
        )
        print(f"xAI Grok client initialized. Default model: {DEFAULT_XAI_MODEL}")
    except Exception as e:
        print(f"Warning: Failed to initialize xAI Grok client: {e}")
        xai_client = None
else:
    print("Warning: XAI_API_KEY not found in .env. xAI Grok features disabled.")


# --- FastAPI App ---
app = FastAPI(
    title="Multi-LLM Chat & Image API",
    description="Interface for OpenAI (Chat & Image), Google Gemini (Chat), and xAI Grok (Chat).",
    version="1.0.0",
)

# --- Pydantic Models ---
class ModelProvider(str, Enum):
    """Enum for selecting the LLM provider."""
    OPENAI = "openai"
    GEMINI = "gemini"
    XAI = "xai" # Using 'xai' for clarity with Grok

class ChatMessage(BaseModel):
    """Structure for a single message in the conversation history."""
    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant', 'system')")
    content: str = Field(..., description="Content of the message")

    # Allow extra fields for potential future use or provider specifics
    class Config:
        extra = "allow"

class ChatRequest(BaseModel):
    model_provider: ModelProvider = Field(..., description="The LLM provider to use (openai, gemini, xai)")
    message: str = Field(..., description="The latest user message")
    history: List[ChatMessage] = Field(default_factory=list, description="Conversation history (optional)")
    model_name: Optional[str] = Field(None, description="Specific model name to override the default (optional)")

class ChatResponse(BaseModel):
    response: str
    model_used: str

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="The prompt for image generation (uses OpenAI DALL-E)")
    # Optional: Add size, quality, etc. later if needed

class ImageGenerationResponse(BaseModel):
    image_url: str
    model_used: str = DEFAULT_OPENAI_IMAGE_MODEL # Hardcode DALL-E model used


# --- Helper Function for Gemini History Formatting ---
def format_history_for_gemini(history: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Converts standard history format to Gemini's expected format."""
    gemini_history = []
    for msg in history:
        # Gemini uses 'user' and 'model' roles
        role = "user" if msg.role.lower() == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg.content]})
    return gemini_history

# --- API Endpoints ---

@app.get("/")
async def read_root():
    """Root endpoint providing basic API info."""
    available_services = []
    if openai_client: available_services.extend(["OpenAI Chat", "OpenAI Image"])
    if gemini_model: available_services.append("Gemini Chat")
    if xai_client: available_services.append("xAI Grok Chat")

    return {
        "message": "Welcome to the Multi-LLM API!",
        "available_services": available_services or ["No services configured. Check .env keys."],
        "docs_url": "/docs"
        }

@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest = Body(...)):
    """Handles chat requests, routing to the specified LLM provider."""
    response_text = ""
    model_identifier = ""

    try:
        if request.model_provider == ModelProvider.OPENAI:
            if not openai_client:
                raise HTTPException(status_code=503, detail="OpenAI service not configured or unavailable.")

            chat_model = request.model_name or DEFAULT_OPENAI_CHAT_MODEL
            model_identifier = f"openai_{chat_model}"

            messages_openai = [msg.model_dump(include={'role', 'content'}) for msg in request.history]
            messages_openai.append({"role": "user", "content": request.message})

            completion = openai_client.chat.completions.create(
                model=chat_model,
                messages=messages_openai,
            )
            response_text = completion.choices[0].message.content or "[No content]"

        elif request.model_provider == ModelProvider.GEMINI:
            if not gemini_model:
                 raise HTTPException(status_code=503, detail="Gemini service not configured or unavailable.")

            # Note: Gemini model switching via request.model_name would require re-instantiating
            # genai.GenerativeModel inside the handler if the name differs from the default.
            # For simplicity, we'll use the pre-configured default model for now.
            gemini_chat_model = DEFAULT_GEMINI_MODEL # Or handle request.model_name if needed
            model_identifier = f"gemini_{gemini_chat_model}"


            formatted_history = format_history_for_gemini(request.history)
            # Use the pre-initialized model object
            chat_session = gemini_model.start_chat(history=formatted_history)

            response = chat_session.send_message(request.message)

            if response.text:
                response_text = response.text
            elif response.prompt_feedback.block_reason:
                response_text = f"[Blocked - Reason: {response.prompt_feedback.block_reason.name}]"
            else:
                response_text = "[Received empty response from Gemini]"


        elif request.model_provider == ModelProvider.XAI:
            if not xai_client:
                raise HTTPException(status_code=503, detail="xAI Grok service not configured or unavailable.")

            xai_chat_model = request.model_name or DEFAULT_XAI_MODEL
            model_identifier = f"xai_{xai_chat_model}"

            messages_xai = [msg.model_dump(include={'role', 'content'}) for msg in request.history]
            messages_xai.append({"role": "user", "content": request.message})

            completion = xai_client.chat.completions.create(
                model=xai_chat_model,
                messages=messages_xai,
            )
            response_text = completion.choices[0].message.content or "[No content]"

        else:
            raise HTTPException(status_code=400, detail="Invalid model_provider specified.")

        return ChatResponse(response=response_text.strip(), model_used=model_identifier)

    # --- Specific API Error Handling ---
    except OpenAI_APIError as e:
        provider = "OpenAI" if request.model_provider == ModelProvider.OPENAI else "xAI Grok"
        print(f"Error from {provider} API: {e}")
        raise HTTPException(status_code=e.status_code or 500, detail=f"{provider} API Error: {e.message or e}")
    except (gemini_types.BlockedPromptException, gemini_types.StopCandidateException) as e:
        print(f"Error from Gemini API: {e}")
        raise HTTPException(status_code=400, detail=f"Gemini content policy issue or generation stop: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred processing the request with {request.model_provider.value}.")


@app.post("/generate-image", response_model=ImageGenerationResponse)
async def handle_image_generation(request: ImageGenerationRequest = Body(...)):
    """Handles image generation requests using OpenAI DALL-E."""
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI service not configured or unavailable for image generation.")

    try:
        print(f"Generating image with prompt: '{request.prompt[:50]}...'")
        response = openai_client.images.generate(
            model=DEFAULT_OPENAI_IMAGE_MODEL,
            prompt=request.prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="url"
        )
        image_url = response.data[0].url
        if not image_url:
             raise HTTPException(status_code=500, detail="OpenAI returned success but no image URL.")

        print("Image generation successful.")
        return ImageGenerationResponse(image_url=image_url)

    except OpenAI_APIError as e:
         print(f"OpenAI Image Generation API Error: {e}")
         raise HTTPException(status_code=e.status_code or 500, detail=f"OpenAI Image API Error: {e.message or e}")
    except Exception as e:
        print(f"An unexpected error occurred during image generation: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during image generation.")


# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    print("--- Starting FastAPI Server ---")
    print("API Keys Loaded:")
    print(f"  OpenAI: {'YES' if OPENAI_API_KEY else 'NO'}")
    print(f"  Google: {'YES' if GOOGLE_API_KEY else 'NO'}")
    print(f"  xAI:    {'YES' if XAI_API_KEY else 'NO'}")
    print("-----------------------------")
    print("Access API docs at http://127.0.0.1:8000/docs")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) # Use "main:app" if filename is main.py