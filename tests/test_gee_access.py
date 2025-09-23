# tests/test_openai_key.py

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the .env file
load_dotenv()

# Get your API key securely
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

try:
    # Simple test query
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-4"
        messages=[{"role": "user", "content": "Hello, is my API key working?"}]
    )

    print("✅ Your API key is working!")
    print("🔹 Response:\n", response.choices[0].message.content)

except Exception as e:
    print("❌ Error occurred:", e)