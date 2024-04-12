from dotenv import load_dotenv
load_dotenv()
import os

# wrtie code to read and print antropic api key from .env
# Read the API key from .env file
api_key = os.getenv("ANTHROPIC_API_KEY")

# Print the API key
print(api_key)