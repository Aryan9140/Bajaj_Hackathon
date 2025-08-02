# openai_fix.py - Quick test to verify OpenAI client works
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def test_openai_simple():
    try:
        import openai
        print(f"✅ OpenAI library imported successfully")
        print(f"🔍 OpenAI version: {openai.__version__}")
        
        # Try simple client initialization
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI client created successfully")
        
        # Test embeddings
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=["test"]
        )
        print("✅ OpenAI embeddings API working")
        print(f"📊 Embedding dimension: {len(response.data[0].embedding)}")
        
        # Test chat completions
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        print("✅ OpenAI chat API working")
        print(f"💬 Response: {response.choices[0].message.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {str(e)}")
        return False

if __name__ == "__main__":
    if OPENAI_API_KEY:
        print(f"🔑 OpenAI API Key: {OPENAI_API_KEY[:20]}...")
        test_openai_simple()
    else:
        print("❌ No OpenAI API key found")