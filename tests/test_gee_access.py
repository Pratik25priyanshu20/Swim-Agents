from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-pro",  # Try the most common first
    google_api_key="AIzaSyDDvr5ZXR7mo_2e773VB0PI_ujpqRS0pp4"
)

response = model.invoke("Hello, who are you?")
print(response.content)