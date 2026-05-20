"""账号池管理模块 - 管理多个DeepSeek账号

支持：
- 从JSON文件加载账号密码
- 自动登录获取Token
- Token轮询、健康检查、自动故障转移
"""

import asyncio
import random
import time
import json
import os
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import httpx

from .proxy_adapter import get_proxy_manager


class TokenStatus(Enum):
    """Token状态"""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    RATE_LIMITED = "rate_limited"


@dataclass
class TokenInfo:
    """Token信息"""

    token: str
    status: TokenStatus = TokenStatus.UNKNOWN
    fail_count: int = 0
    success_count: int = 0
    last_used: Optional[str] = None
    last_checked: Optional[str] = None
    error_message: Optional[str] = None
    average_response_time: float = 0.0
    total_requests: int = 0
    expires_at: Optional[int] = None  # Token 过期时间戳
    added_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token[:20] + "..." + self.token[-10:]
            if len(self.token) > 30
            else self.token,
            "status": self.status.value,
            "fail_count": self.fail_count,
            "success_count": self.success_count,
            "last_used": self.last_used,
            "last_checked": self.last_checked,
            "error_message": self.error_message,
            "average_response_time": self.average_response_time,
            "total_requests": self.total_requests,
            "expires_at": self.expires_at,
            "added_at": self.added_at,
        }

    def mark_success(self, response_time: float = 0):
        """标记成功"""
        self.success_count += 1
        self.fail_count = 0
        self.status = TokenStatus.HEALTHY
        self.error_message = None
        self.last_used = datetime.now().isoformat()
        self.total_requests += 1

        # 更新平均响应时间
        if self.average_response_time == 0:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                self.average_response_time * (self.total_requests - 1) + response_time
            ) / self.total_requests

    def mark_fail(self, error: str = ""):
        """标记失败"""
        self.fail_count += 1
        self.error_message = error
        self.last_used = datetime.now().isoformat()

        if self.fail_count >= 3:
            self.status = TokenStatus.UNHEALTHY

    def mark_rate_limited(self):
        """标记速率限制"""
        self.status = TokenStatus.RATE_LIMITED
        self.last_used = datetime.now().isoformat()

    def is_expired(self) -> bool:
        """检查 token 是否过期"""
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at


class AccountInfo:
    """账号信息"""

    def __init__(self, email: str, password: str, token: Optional[str] = None):
        self.email = email
        self.password = password
        self.token = token
        self.last_login: str | None = None
        self.login_error: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email": self.email,
            "password": self.password,
            "token": self.token,
            "last_login": self.last_login,
            "login_error": self.login_error,
        }


class AccountPool:
    """账号池管理器"""

    def __init__(
        self, storage_file: Optional[str] = None, accounts_file: Optional[str] = None
    ):
        self.tokens: Dict[str, TokenInfo] = {}
        self.accounts: Dict[str, AccountInfo] = {}
        self._lock = asyncio.Lock()
        self._current_index = 0
        self.storage_file = storage_file or "account_pool.json"
        self.accounts_file = accounts_file or "accounts.json"
        self._initialized = False
        # 复用 HTTP 客户端
        self._http_client: Optional[httpx.AsyncClient] = None
        # 可配置的版本信息
        self.app_version = os.environ.get("DEEPSEEK_APP_VERSION", "20241129.1")
        self.client_version = os.environ.get("DEEPSEEK_CLIENT_VERSION", "1.8.0")

    async def init(self):
        """初始化账号池"""
        if self._initialized:
            return

        # 初始化 HTTP 客户端
        proxy_manager = get_proxy_manager()
        if not proxy_manager._initialized:
            proxy_manager.init_from_env()
        self._http_client = proxy_manager.create_async_session(use_vless=True)

        # 从环境变量加载Token
        await self._load_from_env()

        # 从JSON文件加载账号
        await self._load_from_accounts_file()

        # 从存储文件加载状态
        await self._load_from_file()

        # 登录所有账号获取Token
        await self._login_all_accounts()

        self._initialized = True
        print(
            f"[AccountPool] Initialized with {len(self.tokens)} tokens, {len(self.accounts)} accounts"
        )

    async def close(self):
        """关闭资源"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._initialized = False

    async def _load_from_env(self):
        """从环境变量加载Token"""
        tokens_str = os.environ.get("DEEPSEEK_TOKENS", "")
        if not tokens_str:
            return

        # 支持多种分隔符
        tokens = []
        for sep in ["\n", ",", ";"]:
            if sep in tokens_str:
                tokens = [t.strip() for t in tokens_str.split(sep) if t.strip()]
                break

        if not tokens:
            tokens = [tokens_str.strip()]

        async with self._lock:
            for token in tokens:
                if token not in self.tokens:
                    self.tokens[token] = TokenInfo(token=token)
                    print(f"[AccountPool] Added token from env: {token[:20]}...")

    async def _load_from_accounts_file(self):
        """从JSON文件加载账号"""
        if not os.path.exists(self.accounts_file):
            return

        try:
            with open(self.accounts_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 支持列表格式
            if isinstance(data, list):
                for account_data in data:
                    email = account_data.get("email")
                    password = account_data.get("password")
                    token = account_data.get("token")

                    if email and password:
                        self.accounts[email] = AccountInfo(email, password, token)
                        if token:
                            async with self._lock:
                                if token not in self.tokens:
                                    self.tokens[token] = TokenInfo(token=token)
                        print(f"[AccountPool] Added account: {email}")

            # 支持字典格式
            elif isinstance(data, dict):
                for email, account_data in data.items():
                    if isinstance(account_data, dict):
                        password = account_data.get("password")
                        token = account_data.get("token")
                        if password:
                            self.accounts[email] = AccountInfo(email, password, token)
                            if token:
                                async with self._lock:
                                    if token not in self.tokens:
                                        self.tokens[token] = TokenInfo(token=token)
                            print(f"[AccountPool] Added account: {email}")

            print(
                f"[AccountPool] Loaded {len(self.accounts)} accounts from {self.accounts_file}"
            )
        except Exception as e:
            print(f"[AccountPool] Failed to load accounts from file: {e}")

    async def _load_from_file(self):
        """从文件加载Token状态"""
        if not os.path.exists(self.storage_file):
            return

        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            async with self._lock:
                for token_data in data.get("tokens", []):
                    token = token_data.get("token")
                    if token and token in self.tokens:
                        # 恢复状态
                        info = self.tokens[token]
                        info.fail_count = token_data.get("fail_count", 0)
                        info.success_count = token_data.get("success_count", 0)
                        info.average_response_time = token_data.get(
                            "average_response_time", 0
                        )
                        info.total_requests = token_data.get("total_requests", 0)

                        # 恢复状态枚举
                        status_str = token_data.get("status", "unknown")
                        try:
                            info.status = TokenStatus(status_str)
                        except ValueError:
                            info.status = TokenStatus.UNKNOWN

            print(
                f"[AccountPool] Loaded {len(data.get('tokens', []))} tokens from file"
            )
        except Exception as e:
            print(f"[AccountPool] Failed to load from file: {e}")

    async def _login_all_accounts(self):
        """登录所有账号获取Token"""
        if not self.accounts:
            return

        print(f"[AccountPool] Logging in {len(self.accounts)} accounts...")

        for email, account in self.accounts.items():
            if not account.token:
                try:
                    token, expires_at = await self._login_account(
                        email, account.password
                    )
                    if token:
                        account.token = token
                        account.last_login = datetime.now().isoformat()
                        account.login_error = None

                        async with self._lock:
                            if token not in self.tokens:
                                self.tokens[token] = TokenInfo(
                                    token=token, expires_at=expires_at
                                )
                        print(f"[AccountPool] Login successful: {email}")
                except Exception as e:
                    account.login_error = str(e)
                    print(f"[AccountPool] Login failed for {email}: {e}")

    async def _login_account(
        self, email: str, password: str
    ) -> tuple[Optional[str], Optional[int]]:
        """登录单个账号获取Token (使用复用的 http client)"""
        if self._http_client is None:
            raise RuntimeError("AccountPool not initialized")

        url = "https://chat.deepseek.com/api/v0/users/login"

        headers = {
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,de-DE;q=0.5,de;q=0.4",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "sec-ch-ua": '"Microsoft Edge";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "x-app-version": self.app_version,
            "x-client-locale": "zh_CN",
            "x-client-platform": "web",
            "x-client-timezone-offset": "28800",
            "x-client-version": self.client_version,
            "referrer": "https://chat.deepseek.com/sign_in",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36 Edg/147.0.0.0",
        }

        data = {
            "email": email,
            "mobile": "",
            "password": password,
            "area_code": "",
            "device_id": "",
            "os": "web",
        }

        try:
            response = await self._http_client.post(url, headers=headers, json=data)

            if response.status_code != 200:
                raise Exception(
                    f"登录失败: HTTP {response.status_code}, {response.text}"
                )

            result = response.json()

            if result.get("code") != 0:
                error_msg = (
                    result.get("msg")
                    or result.get("data", {}).get("biz_msg")
                    or "Unknown error"
                )
                raise Exception(f"登录失败: {error_msg}")

            # 提取 Token
            biz_data = result.get("data", {}).get("biz_data", {})
            user = biz_data.get("user", {})
            token = user.get("token")

            if not token:
                raise Exception("登录成功但未获取到 Token")

            # 提取过期时间 (示例，实际过期时间可能在其他字段)
            expires_at = user.get("expires_at") or user.get("token_expires_at")
            if expires_at:
                expires_at = int(expires_at)
            else:
                expires_at = int(time.time()) + 7 * 24 * 3600  # 默认7天

            return token, expires_at

        except httpx.HTTPError as e:
            raise Exception(f"HTTP 错误: {e}") from e

    async def get_token(self, strategy: str = "round_robin") -> Optional[str]:
        """获取一个可用的Token (线程/协程安全)"""
        async with self._lock:
            # 过滤健康的且未过期的 token
            healthy_tokens = [
                t
                for t, info in self.tokens.items()
                if info.status == TokenStatus.HEALTHY and not info.is_expired()
            ]

            if not healthy_tokens:
                # 如果没有健康token，尝试使用未知状态的
                healthy_tokens = [
                    t
                    for t, info in self.tokens.items()
                    if info.status == TokenStatus.UNKNOWN and not info.is_expired()
                ]

            if not healthy_tokens:
                return None

            if strategy == "random":
                return random.choice(healthy_tokens)
            else:  # round_robin
                token = healthy_tokens[self._current_index % len(healthy_tokens)]
                self._current_index += 1
                return token

    async def mark_token_success(self, token: str, response_time: float = 0):
        """标记Token使用成功 (线程/协程安全)"""
        async with self._lock:
            if token in self.tokens:
                self.tokens[token].mark_success(response_time)
                await self._save_state_async()

    async def mark_token_fail(self, token: str, error: str = ""):
        """标记Token使用失败 (线程/协程安全)"""
        async with self._lock:
            if token in self.tokens:
                # 如果是认证错误（401/403），标记为过期
                if "401" in error or "403" in error or "invalid token" in error.lower():
                    self.tokens[token].expires_at = 0
                    self.tokens[token].status = TokenStatus.UNHEALTHY
                else:
                    self.tokens[token].mark_fail(error)
                await self._save_state_async()

    async def _save_state_async(self):
        """保存状态到文件 (异步)"""
        try:
            data = {"tokens": [info.to_dict() for info in self.tokens.values()]}
            # 在线程池中执行文件写入，避免阻塞事件循环
            await asyncio.to_thread(
                lambda: json.dump(
                    data,
                    open(self.storage_file, "w", encoding="utf-8"),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        except Exception as e:
            print(f"[AccountPool] Failed to save state: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """获取账号池统计信息"""
        async with self._lock:
            return {
                "total_tokens": len(self.tokens),
                "healthy_tokens": sum(
                    1
                    for info in self.tokens.values()
                    if info.status == TokenStatus.HEALTHY and not info.is_expired()
                ),
                "unhealthy_tokens": sum(
                    1
                    for info in self.tokens.values()
                    if info.status == TokenStatus.UNHEALTHY
                ),
                "total_accounts": len(self.accounts),
                "token_details": [info.to_dict() for info in self.tokens.values()],
                "account_details": [acc.to_dict() for acc in self.accounts.values()],
            }
