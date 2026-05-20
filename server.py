"""OpenAI Compatible API Server for DeepSeek AI

This is the main API server - users provide their own tokens.
For account pool functionality, see the pool/ subdirectory.
"""

import os
import json
import time
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# 导入 deepseek模块
from deepseek_ai import DeepSeekClient
from deepseek_ai.proxy_adapter import init_proxy_manager
from deepseek_ai.subscription import init_subscriptions_from_env
from deepseek_ai.node_storage import init_node_storage
from deepseek_ai.node_tester import init_node_tester

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global settings from environment
AUTO_DELETE_SESSION = os.environ.get("AUTO_DELETE_SESSION", "false").lower() == "true"

# Global managers
proxy_manager = None
subscription_manager = None
node_storage = None
node_tester = None


def init_services():
    """Initialize services on startup"""
    global proxy_manager, subscription_manager, node_storage, node_tester

    # Initialize proxy manager
    try:
        proxy_manager = init_proxy_manager()
        logger.info("[Server] Proxy manager initialized")
    except Exception as e:
        logger.error(f"[Server] Failed to initialize proxy manager: {e}")
        proxy_manager = None

    # Initialize subscription manager
    try:
        subscription_manager = init_subscriptions_from_env()
        logger.info("[Server] Subscription manager initialized")
    except Exception as e:
        logger.error(f"[Server] Failed to initialize subscription manager: {e}")
        subscription_manager = None

    # Initialize node storage
    try:
        node_storage = init_node_storage()
        logger.info("[Server] Node storage initialized")
    except Exception as e:
        logger.error(f"[Server] Failed to initialize node storage: {e}")
        node_storage = None

    # Initialize node tester
    try:
        node_tester = init_node_tester()
        logger.info("[Server] Node tester initialized")
    except Exception as e:
        logger.error(f"[Server] Failed to initialize node tester: {e}")
        node_tester = None


# Supported models
SUPPORTED_MODELS = [
    "deepseek-v4-flash",
    "deepseek-v4-flash-think",
    "deepseek-v4-flash-fast",
    "deepseek-v4-pro",
    "deepseek-v4-pro-think",
    "deepseek-v4-pro-fast",
]


def get_auth_token() -> str:
    """Get authorization token from env"""
    auth_header = os.environ.get("DEEPSEEK_TOKENS")
    if not auth_header:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "No AUTH_TOKEN provided in environment variables",
                    "type": "authentication_error",
                }
            },
        )
    return auth_header


def select_random_token(token_string: str) -> str:
    """Select a random token from comma-separated list"""
    import random

    tokens = [t.strip() for t in token_string.split(",") if t.strip()]
    if not tokens:
        raise ValueError("No valid tokens provided")
    return random.choice(tokens)


@app.get("/v1/models")
async def list_models():
    """List available models"""
    models = [
        {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "deepseek-ai",
        }
        for model_id in SUPPORTED_MODELS
    ]
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Chat completions endpoint - users provide their own tokens"""
    try:
        # Get token
        token_string = get_auth_token()
        token = select_random_token(token_string)

        # Parse request
        try:
            data = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "Invalid JSON body",
                        "type": "invalid_request_error",
                    }
                },
            )

        if not data:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "Empty JSON body",
                        "type": "invalid_request_error",
                    }
                },
            )

        model = data.get("model", "deepseek-chat")
        messages = data.get("messages", [])
        stream = data.get("stream", False)
        temperature = data.get("temperature")
        web_search = data.get("web_search", False)
        reasoning_effort = data.get("reasoning_effort")
        thinking = data.get("thinking")  # OpenAI compatible format: {"type": "enabled"}
        tools = data.get("tools")

        # Create client
        client = DeepSeekClient(token=token, use_proxy=True)

        # Determine thinking mode from model name suffix or API parameters
        model_lower = model.lower()

        # Method 1: Model name suffix (-think or -fast)
        if "-think" in model_lower:
            thinking_enabled = True
        elif "-fast" in model_lower:
            thinking_enabled = False
        else:
            # Method 2: API parameters (thinking or reasoning_effort)
            if (
                thinking
                and isinstance(thinking, dict)
                and thinking.get("type") == "enabled"
            ):
                thinking_enabled = True
            elif reasoning_effort and reasoning_effort.lower() in [
                "low",
                "medium",
                "high",
            ]:
                thinking_enabled = True
            else:
                # Default: flash = no thinking, pro = thinking
                thinking_enabled = "pro" in model_lower

        result = await client.chat_completions(
            model=model,
            messages=messages,
            stream=stream,
            temperature=temperature,
            web_search=web_search,
            reasoning_effort=reasoning_effort,
            thinking_enabled=thinking_enabled,
            tools=tools,
            auto_delete_session=AUTO_DELETE_SESSION,
        )

        if isinstance(result, AsyncGenerator):
            print("Streaming response initiated")

            # Streaming response
            async def generate():
                try:
                    async for chunk in result:
                        yield chunk
                except Exception as e:
                    logger.error(f"[Server] Stream error: {e}")
                    error_chunk = json.dumps(
                        {"error": {"message": str(e), "type": "internal_error"}}
                    )
                    yield f"data: {error_chunk}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        if isinstance(result, dict):
            # Non-streaming response
            return JSONResponse(content=result)

    except HTTPException:
        raise
    # except Exception as e:
    #     logger.error(f"[Server] Error: {e}")
    #     # Simple heuristic to return 401 if it's token/auth related string
    #     if isinstance(e, ValueError) and (
    #         "auth" in str(e).lower() or "token" in str(e).lower()
    #     ):
    #         return JSONResponse(
    #             status_code=401,
    #             content={"error": {"message": str(e), "type": "authentication_error"}},
    #         )
    #     return JSONResponse(
    #         status_code=500,
    #         content={"error": {"message": str(e), "type": "internal_error"}},
    #     )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "deepseek-ai-openai-api",
        "version": "1.0.0",
    }


@app.get("/v1/proxy/stats")
async def proxy_stats():
    """Get proxy statistics"""
    global proxy_manager

    if proxy_manager is None:
        return {"enabled": False, "message": "Proxy manager not initialized"}

    try:
        stats = proxy_manager.get_stats()
        return {"enabled": True, "stats": stats}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}},
        )


@app.post("/v1/nodes/test")
async def test_nodes():
    """Test all nodes"""
    global node_tester, node_storage

    if node_tester is None or node_storage is None:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "Node tester or storage not initialized",
            },
        )

    try:
        results = node_tester.test_all_nodes()
        return {"success": True, "results": results}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}},
        )


@app.get("/v1/nodes/stats")
async def nodes_stats():
    """Get nodes statistics"""
    global node_storage

    if node_storage is None:
        return {"enabled": False, "message": "Node storage not initialized"}

    try:
        stats = node_storage.get_stats()
        return {"enabled": True, "stats": stats}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}},
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "DeepSeek AI OpenAI Compatible API",
        "version": "1.0.0",
        "description": "Users provide their own tokens for API access",
        "features": [
            "openai_compatible_api",
            "streaming_support",
            "proxy_support",
            "web_search",
            "reasoning_mode",
            "tool_calls",
        ],
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
            "proxy_stats": "/v1/proxy/stats",
            "nodes_test": "/v1/nodes/test",
            "nodes_stats": "/v1/nodes/stats",
        },
        "account_pool": {
            "description": "For account pool functionality, see /pool/ subdirectory",
            "endpoint": "/pool/",
        },
    }


if __name__ == "__main__":
    init_services()

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting DeepSeek AI OpenAI API Server (FastAPI) on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
