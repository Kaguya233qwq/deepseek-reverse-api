"""Start server script with environment loading"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import and run the server
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get('PORT', 8000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print(f"Starting DeepSeek AI Reverse API Server on {host}:{port}")
    print(f"Features: Account Pool, Vless Proxy Support, OpenAI Compatible API")
    
    uvicorn.run(
        "server:asgi_app",
        host=host,
        port=port,
        reload=False,
        log_level=os.environ.get('LOG_LEVEL', 'info').lower()
    )
