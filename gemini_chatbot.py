import os
import google.generativeai as genai
from google.generativeai.types import generation_types # For specific exceptions
from dotenv import load_dotenv
import sys

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") # Or os.getenv("GEMINI_API_KEY")
MODEL_NAME = 'gemini-2.0-flash' # Using a standard, reliable model


if not API_KEY:
    print("Error: GOOGLE_API_KEY (or GEMINI_API_KEY) not found in .env file.")
    print("Please create a .env file with your Google API Key.")
    sys.exit(1)

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"Successfully initialized Gemini model: {MODEL_NAME}")
    chat_session = model.start_chat(history=[]) # Start with empty history

except Exception as e:
    print(f"Error configuring or initializing Gemini client: {e}")
    sys.exit(1)

print("-" * 30)
print("Gemini Terminal Chat (Simple Version)")
print(f"Model: {MODEL_NAME}")
print("Type 'quit' or 'exit' to end the chat.")
print("-" * 30)


while True:
    try:
        user_input = input("You: ")
        cleaned_input = user_input.strip()

        if cleaned_input.lower() in ["quit", "exit"]:
            print("Gemini: Goodbye!")
            break

        if not cleaned_input:
            continue


        print("Gemini: Thinking...")
        response = chat_session.send_message(cleaned_input)
        print(f"Gemini: {response.text}")

        if not response.text and response.prompt_feedback.block_reason:
             print(f"Gemini: [Blocked - Reason: {response.prompt_feedback.block_reason.name}]")
        elif not response.text and not response.candidates:
             print(f"Gemini: [Received an empty response without explicit blocking]")



    except generation_types.BlockedPromptException as e:
        print(f"Gemini: [Blocked - Your prompt was blocked by safety settings before sending.]")
        print(f"         Reason: {e}")
    except generation_types.StopCandidateException as e:
         print(f"Gemini: [Response stopped unexpectedly] - {e}")

    except KeyboardInterrupt:
        print("\nGemini: Goodbye! (Interrupted by user)")
        break
    except Exception as e:
        print(f"\nAn unexpected error occurred: {type(e).__name__} - {e}")
        print("Exiting due to unexpected error.")
        break