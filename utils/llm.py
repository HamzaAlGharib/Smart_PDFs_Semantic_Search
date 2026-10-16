from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("gemma_api_key")

if api_key is None:
    raise ValueError("key not found...")



    
