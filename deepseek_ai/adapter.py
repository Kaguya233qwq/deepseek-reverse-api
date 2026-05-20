"""DeepSeek AI Adapter for chat.deepseek.com - Based on Chat2API logic"""

import uuid
import time
from typing import Dict, Optional, Tuple, List
import re

import httpx

from .proxy_adapter import get_proxy_manager, init_proxy_manager
from .pow_solver import calculate_challenge_answer


class DeepSeekAPIError(Exception):
    """DeepSeek API 基础异常"""

    pass


class DeepSeekAuthError(DeepSeekAPIError):
    """认证相关错误（Token无效/过期）"""

    pass


class DeepSeekRequestError(DeepSeekAPIError):
    """请求错误"""

    pass


class DeepSeekAdapter:
    """DeepSeek AI Adapter for chat.deepseek.com - httpx based"""

    DEEPSEEK_API_BASE = "https://chat.deepseek.com/api"

    DEFAULT_HEADERS = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "Origin": "https://chat.deepseek.com",
        "Referer": "https://chat.deepseek.com/",
        "Sec-Ch-Ua": '"Microsoft Edge";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36 Edg/147.0.0.0",
        "X-App-Version": "20241129.1",
        "X-Client-Locale": "zh_CN",
        "X-Client-Platform": "web",
        "X-Client-Version": "1.8.0",
        "X-Client-Timezone-Offset": "28800",
    }

    MODEL_ALIASES = {
        "deepseek-v4-flash": "deepseek-chat",
        "deepseek-v4-pro": "deepseek-reasoner",
    }

    # 可配置的超时/过期时间
    TOKEN_EXPIRY_SECONDS = 3600  # 1小时
    SESSION_EXPIRY_SECONDS = 300  # 5分钟

    def __init__(self, token: str, use_proxy: bool = True):
        """Initialize DeepSeek Adapter

        Args:
            token: DeepSeek API token
            use_proxy: Whether to use proxy (VLess or HTTP)
            async_mode: Whether to use async mode (httpx.AsyncClient)
        """
        self.token = token
        self._access_token: Optional[str] = None
        self._token_expires_at: int = 0
        self._session_id: str = ""
        self._session_created_at: float = 0
        self.use_proxy = use_proxy

        # 异步锁，保护共享状态
        self._async_lock = None  # 会在首次异步调用中延迟初始化，或由框架保证事件循环

        # Initialize proxy manager if needed
        if use_proxy:
            self.proxy_manager = get_proxy_manager()
            if (
                not hasattr(self.proxy_manager, "_initialized")
                or not self.proxy_manager._initialized
            ):
                self.proxy_manager = init_proxy_manager()
                self.proxy_manager._initialized = True
            # Create httpx client via proxy manager
            self.client = self.proxy_manager.create_client(use_vless=True)
        else:
            # Direct client without proxy

            self.client = httpx.AsyncClient(timeout=120.0)

    def _uuid(self) -> str:
        """Generate UUID"""
        return str(uuid.uuid4())

    def get_headers(self, extra_headers: Optional[Dict] = None) -> Dict[str, str]:
        """Get request headers"""
        headers = self.DEFAULT_HEADERS.copy()
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def map_model(self, openai_model: str) -> str:
        """Map OpenAI model name to DeepSeek model name"""
        model = openai_model.lower()

        # Remove suffixes for mapping
        base_model = model.replace("-think", "").replace("-fast", "")

        if base_model in self.MODEL_ALIASES:
            return self.MODEL_ALIASES[base_model]

        return base_model

    # --- Sync methods ---
    async def _acquire_token_async(self) -> str | None:
        """Acquire access token from DeepSeek (sync, thread-safe)"""
        if not self.token:
            raise DeepSeekAuthError("DeepSeek token not configured")

        if self._async_lock is None:
            import asyncio

            self._async_lock = asyncio.Lock()

        async with self._async_lock:
            if self._access_token and self._token_expires_at > int(time.time()):
                return self._access_token

            url = f"{self.DEEPSEEK_API_BASE}/v0/users/current"
            response = await self.client.get(
                url,
                headers={"Authorization": f"Bearer {self.token}", **self.get_headers()},
            )

            if response.status_code in [401, 403]:
                raise DeepSeekAuthError(
                    "Token invalid or expired, please get a new token"
                )

            if response.status_code != 200:
                raise DeepSeekRequestError(
                    f"Failed to acquire token: HTTP {response.status_code}"
                )

            data = response.json()
            biz_data = data.get("data", {}).get("biz_data") or data.get("biz_data")

            if not biz_data or not biz_data.get("token"):
                error_msg = (
                    data.get("msg")
                    or data.get("data", {}).get("biz_msg")
                    or "Unknown error"
                )
                raise DeepSeekRequestError(f"Failed to acquire token: {error_msg}")

            self._access_token = biz_data["token"]
            self._token_expires_at = int(time.time()) + self.TOKEN_EXPIRY_SECONDS

            return self._access_token

    async def _create_session_async(self) -> str:
        """Create a new chat session (sync, thread-safe)"""
        if self._async_lock is None:
            import asyncio

            self._async_lock = asyncio.Lock()

        async with self._async_lock:
            if (
                self._session_id
                and (time.time() - self._session_created_at)
                < self.SESSION_EXPIRY_SECONDS
            ):
                return self._session_id

        token = await self._acquire_token_async()

        url = f"{self.DEEPSEEK_API_BASE}/v0/chat_session/create"
        response = await self.client.post(
            url,
            json={"character_id": None},
            headers={
                "Authorization": f"Bearer {token}",
                **self.get_headers(),
            },
        )

        data = response.json()
        biz_data = data.get("data", {}).get("biz_data") or data.get("biz_data")

        if response.status_code != 200 or not biz_data:
            raise DeepSeekRequestError(
                f"Failed to create session: {data.get('msg') or response.status_code}"
            )

        if "chat_session" in biz_data and isinstance(biz_data["chat_session"], dict):
            session_id = biz_data["chat_session"].get("id")
        else:
            session_id = biz_data.get("id")

        if not session_id:
            raise DeepSeekRequestError(
                "Failed to create session: no session id in response"
            )

        async with self._async_lock:
            self._session_id = session_id
            self._session_created_at = time.time()

        return self._session_id

    async def _get_challenge_async(self, target_path: str) -> Dict:
        """Get POW challenge from DeepSeek (sync)"""
        token = await self._acquire_token_async()
        url = f"{self.DEEPSEEK_API_BASE}/v0/chat/create_pow_challenge"
        response = await self.client.post(
            url,
            json={"target_path": target_path},
            headers={
                "Authorization": f"Bearer {token}",
                **self.get_headers(),
            },
        )

        data = response.json()
        biz_data = data.get("data", {}).get("biz_data") or data.get("biz_data")

        if response.status_code != 200 or not biz_data or not biz_data.get("challenge"):
            raise DeepSeekRequestError(
                f"Failed to get challenge: {data.get('msg') or response.status_code}"
            )

        return biz_data["challenge"]

    def _calculate_challenge_answer(self, challenge: Dict) -> str:
        """Calculate challenge answer using DeepSeekHashV1 WASM"""
        try:
            answer = calculate_challenge_answer(challenge)
            return answer
        except Exception as e:
            raise DeepSeekRequestError(f"Failed to calculate challenge answer: {e}")

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages to DeepSeek prompt format (more robust)"""
        processed_messages = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "assistant" and message.get("tool_calls"):
                tool_calls_text = []
                for tc in message["tool_calls"]:
                    func = tc.get("function", {})
                    tool_calls_text.append(
                        f"<tool_calling>\n<name>{func.get('name', '')}</name>\n<arguments>{func.get('arguments', '')}</arguments>\n</tool_calling>"
                    )
                text = "\n".join(tool_calls_text)
            elif role == "tool" and message.get("tool_call_id"):
                text = f'<tool_response tool_call_id="{message["tool_call_id"]}">\n{content}\n</tool_response>'
            elif isinstance(content, list):
                texts = [
                    item.get("text", "")
                    for item in content
                    if item.get("type") == "text"
                ]
                text = "\n".join(texts)
            else:
                text = str(content or "")

            processed_messages.append({"role": role, "text": text})

        if not processed_messages:
            return ""

        merged_blocks = []
        current_block = {**processed_messages[0]}

        for i in range(1, len(processed_messages)):
            msg = processed_messages[i]
            if msg["role"] == current_block["role"]:
                current_block["text"] += f"\n\n{msg['text']}"
            else:
                merged_blocks.append(current_block)
                current_block = {**msg}
        merged_blocks.append(current_block)

        result = []
        for index, block in enumerate(merged_blocks):
            if block["role"] == "assistant":
                # 转义潜在的分隔符
                safe_text = block["text"].replace("<｜", "&lt;｜")
                result.append(f"<｜Assistant｜>{safe_text}<｜end of sentence｜>")
            elif block["role"] in ["user", "system"]:
                safe_text = block["text"].replace("<｜", "&lt;｜")
                result.append(f"<｜User｜>{safe_text}" if index > 0 else safe_text)
            elif block["role"] == "tool":
                safe_text = block["text"].replace("<｜", "&lt;｜")
                result.append(f"<｜User｜>{safe_text}")

        prompt = "".join(result)

        # 移除 Markdown 图片，但避免误删其他内容
        prompt = re.sub(r"!\[.*?\]\(.*?\)", "", prompt)
        # 移除可能残留的 HTML 标签
        prompt = re.sub(r"<[^>]+>", "", prompt)

        return prompt

    async def chat_completion_async(
        self,
        model: str,
        messages: list,
        stream: bool = True,
        temperature: Optional[float] = None,
        web_search: bool = False,
        reasoning_effort: Optional[str] = None,
        thinking_enabled: Optional[bool] = None,
    ) -> Tuple[httpx.Response, str]:
        """Send chat completion request (sync)"""
        token = await self._acquire_token_async()
        session_id = await self._create_session_async()

        challenge = await self._get_challenge_async("/api/v0/chat/completion")
        challenge_answer = self._calculate_challenge_answer(challenge)

        prompt = self._messages_to_prompt(messages)

        # 构建请求体
        payload = {
            "chat_session_id": session_id,
            "parent_message_id": 0,
            "prompt": prompt,
            "model": self.map_model(model),
            "stream": stream,
            "search_enabled": web_search,
            "ref_file_ids": [],
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if thinking_enabled is not None:
            payload["thinking_enabled"] = thinking_enabled

        # 添加工具/函数调用支持（如果需要）
        # 根据您的原始代码，这里可能还有其他逻辑，但基础版本先保持简单

        headers = {
            "Authorization": f"Bearer {token}",
            "X-Ds-Pow-Response": challenge_answer,
            **self.get_headers(),
        }

        url = f"{self.DEEPSEEK_API_BASE}/v0/chat/completion"

        response = await self.client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            print(f"DEBUG RESPONSE: {response.text}")
            # 尝试解析错误信息
            try:
                error_data = response.json()
                error_msg = (
                    error_data.get("msg") or error_data.get("error") or "Unknown error"
                )
            except Exception:
                error_msg = f"HTTP {response.status_code}"
            raise DeepSeekRequestError(f"Chat completion failed: {error_msg}")

        return response, session_id

    async def close(self):
        """关闭 HTTP 客户端"""
        if hasattr(self, "client") and self.client:
            await self.client.aclose()

    async def delete_session_async(self, session_id: str) -> bool:
        """Delete a chat session (async)"""
        try:
            token = await self._acquire_token_async()
            url = f"{self.DEEPSEEK_API_BASE}/v0/chat_session/delete"
            response = await self.client.post(
                url,
                json={"chat_session_id": session_id},
                headers={
                    "Authorization": f"Bearer {token}",
                    **self.get_headers(),
                },
            )
            data = response.json()
            success = response.status_code == 200 and data.get("code") == 0
            if success and self._session_id == session_id:
                self._session_id = ""
            return success
        except Exception:
            return False
