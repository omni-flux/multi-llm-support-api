import os
from openai import OpenAI, APIError
from dotenv import load_dotenv
import sys

# --- Configuration ---
load_dotenv()

API_KEY = os.getenv("XAI_API_KEY")
BASE_URL = "https://api.x.ai/v1"
MODEL_NAME = "grok-3-beta"

# --- Initialization ---
if not API_KEY:
    print("Error: XAI_API_KEY not found in .env file.")
    print("Please create a .env file with your xAI API Key.")
    sys.exit(1)

try:

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    print(f"Successfully initialized client for xAI Grok.")
    print(f"Using Model: {MODEL_NAME} via endpoint: {BASE_URL}")
except APIError as e:
     print(f"xAI API Error during initialization: {e}")
     sys.exit(1)
except Exception as e:
    print(f"Error initializing client for xAI: {e}")
    sys.exit(1)


conversation_history = [
     {"role": "system", "content": "You are Grok, a helpful AI from xAI."}
]

print("-" * 30)
print("xAI Grok Terminal Chat")
print(f"Model: {MODEL_NAME}")
print("Type 'quit' or 'exit' to end the chat.")
print("-" * 30)


while True:
    try:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print("Grok: Goodbye!")
            break

        if not user_input:
            continue

        conversation_history.append({"role": "user", "content": user_input})

        print("Grok: Thinking...")
        try:
            chat_completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation_history,
            )

            if chat_completion.choices and chat_completion.choices[0].message:
                 ai_response_content = chat_completion.choices[0].message.content
            else:
                 ai_response_content = None

            if ai_response_content:
                conversation_history.append({"role": "assistant", "content": ai_response_content})
                print(f"Grok: {ai_response_content.strip()}")
            else:
                print("Grok: [Received an empty or unexpected response format]")
                if conversation_history and conversation_history[-1]["role"] == "user":
                    conversation_history.pop()


        except APIError as e:
            print(f"Grok: Error communicating with xAI API: {e}")
            if conversation_history and conversation_history[-1]["role"] == "user":
                conversation_history.pop()
        except Exception as e:
            print(f"Grok: An unexpected error occurred during chat: {e}")
            if conversation_history and conversation_history[-1]["role"] == "user":
                 conversation_history.pop()

    except KeyboardInterrupt:
        print("\nGrok: Goodbye! (Interrupted by user)")
        break
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main loop: {type(e).__name__} - {e}")
        break