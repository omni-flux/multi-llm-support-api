import os
from openai import OpenAI, APIError
from dotenv import load_dotenv
import sys

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = "gpt-4o"
IMAGE_MODEL = "dall-e-3"

if not API_KEY:
    print("Error: OPENAI_API_KEY not found in .env file.")
    print("Please create a .env file with your OpenAI API Key.")
    sys.exit(1)

try:
    client = OpenAI()
    print(f"Successfully initialized OpenAI client.")
    print(f"Using Chat Model: {CHAT_MODEL}, Image Model: {IMAGE_MODEL}")
except APIError as e:
     print(f"OpenAI API Error during initialization: {e}")
     sys.exit(1)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    sys.exit(1)

conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("-" * 30)
print("OpenAI Terminal Chat")
print("Type '/image <prompt>' to generate an image.")
print("Type 'quit' or 'exit' to end the chat.")
print("-" * 30)

while True:
    try:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print("AI: Goodbye!")
            break

        if not user_input:
            continue

        # --- Image Generation Handling ---
        if user_input.lower().startswith("/image "):
            prompt = user_input[len("/image "):].strip()
            if not prompt:
                print("AI: Please provide a prompt after /image.")
                continue

            print(f"AI: Generating image for prompt: \"{prompt}\"...")
            try:
                response = client.images.generate(
                    model=IMAGE_MODEL,
                    prompt=prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                    response_format="url"
                )
                image_url = response.data[0].url
                print(f"AI: Image generated! You can view it here:\n{image_url}")

            except APIError as e:
                 print(f"AI: Error generating image: {e}")
            except Exception as e:
                 print(f"AI: An unexpected error occurred during image generation: {e}")

            continue


        conversation_history.append({"role": "user", "content": user_input})

        print("AI: Thinking...")
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=conversation_history,
            )

            ai_response_content = response.choices[0].message.content

            if ai_response_content:
                conversation_history.append({"role": "assistant", "content": ai_response_content})
                print(f"AI: {ai_response_content.strip()}")
            else:
                print("AI: [Received an empty response]")


        except APIError as e:
            print(f"AI: Error communicating with OpenAI: {e}")
            conversation_history.pop()
        except Exception as e:
            print(f"AI: An unexpected error occurred during chat: {e}")
            if conversation_history and conversation_history[-1]["role"] == "user":
                 conversation_history.pop()



    except KeyboardInterrupt:
        print("\nAI: Goodbye! (Interrupted by user)")
        break
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main loop: {type(e).__name__} - {e}")
        break
