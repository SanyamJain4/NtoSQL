import os
from dotenv import load_dotenv

load_dotenv()
print("GEMINI_API_KEY=", os.getenv("GEMINI_API_KEY"))
try:
    import google.genai as genai
    print("google.genai imported, version:", getattr(genai, '__version__', 'unknown'))
except Exception as e:
    print("IMPORT_ERROR:", type(e).__name__, e)
