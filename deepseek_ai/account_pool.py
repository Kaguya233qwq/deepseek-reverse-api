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
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum

import httpx

from .client import DeepSeekClient
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
    added_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'token': self.token[:20] + '...' + self.token[-10:] if len(self.token) > 30 else self.token,
            'status': self.status.value,
            'fail_count': self.fail_count,
            'success_count': self.success_count,
            'last_used': self.last_used,
            'last_checked': self.last_checked,
            'error_message': self.error_message,
            'average_response_time': self.average_response_time,
            'total_requests': self.total_requests,
            'added_at': self.added_at,
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


class AccountInfo:
    """账号信息"""

    def __init__(self, email: str, password: str, token: Optional[str] = None):
        self.email = email
        self.password = password
        self.token = token
        self.last_login = None
        self.login_error = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'email': self.email,
            'password': self.password,
            'token': self.token,
            'last_login': self.last_login,
            'login_error': self.login_error
        }


class AccountPool:
    """账号池管理器"""

    def __init__(self, storage_file: Optional[str] = None, accounts_file: Optional[str] = None):
        self.tokens: Dict[str, TokenInfo] = {}
        self.accounts: Dict[str, AccountInfo] = {}
        self._lock = asyncio.Lock()
        self._current_index = 0
        self.storage_file = storage_file or "account_pool.json"
        self.accounts_file = accounts_file or "accounts.json"
        self._initialized = False

    async def init(self):
        """初始化账号池"""
        if self._initialized:
            return

        # 从环境变量加载Token
        await self._load_from_env()

        # 从JSON文件加载账号
        await self._load_from_accounts_file()

        # 从存储文件加载状态
        await self._load_from_file()

        # 登录所有账号获取Token
        await self._login_all_accounts()

        self._initialized = True
        print(f"[AccountPool] Initialized with {len(self.tokens)} tokens, {len(self.accounts)} accounts")

    async def _load_from_env(self):
        """从环境变量加载Token"""
        tokens_str = os.environ.get('DEEPSEEK_TOKENS', '')
        if not tokens_str:
            return

        # 支持多种分隔符
        tokens = []
        for sep in ['\n', ',', ';']:
            if sep in tokens_str:
                tokens = [t.strip() for t in tokens_str.split(sep) if t.strip()]
                break

        if not tokens:
            tokens = [tokens_str.strip()]

        for token in tokens:
            if token not in self.tokens:
                self.tokens[token] = TokenInfo(token=token)
                print(f"[AccountPool] Added token from env: {token[:20]}...")

    async def _load_from_accounts_file(self):
        """从JSON文件加载账号"""
        if not os.path.exists(self.accounts_file):
            return

        try:
            with open(self.accounts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 支持列表格式
            if isinstance(data, list):
                for account_data in data:
                    email = account_data.get('email')
                    password = account_data.get('password')
                    token = account_data.get('token')

                    if email and password:
                        self.accounts[email] = AccountInfo(email, password, token)
                        if token:
                            if token not in self.tokens:
                                self.tokens[token] = TokenInfo(token=token)
                        print(f"[AccountPool] Added account: {email}")

            # 支持字典格式
            elif isinstance(data, dict):
                for email, account_data in data.items():
                    if isinstance(account_data, dict):
                        password = account_data.get('password')
                        token = account_data.get('token')
                        if password:
                            self.accounts[email] = AccountInfo(email, password, token)
                            if token:
                                if token not in self.tokens:
                                    self.tokens[token] = TokenInfo(token=token)
                            print(f"[AccountPool] Added account: {email}")

            print(f"[AccountPool] Loaded {len(self.accounts)} accounts from {self.accounts_file}")
        except Exception as e:
            print(f"[AccountPool] Failed to load accounts from file: {e}")

    async def _load_from_file(self):
        """从文件加载Token状态"""
        if not os.path.exists(self.storage_file):
            return

        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for token_data in data.get('tokens', []):
                token = token_data.get('token')
                if token and token in self.tokens:
                    # 恢复状态
                    info = self.tokens[token]
                    info.fail_count = token_data.get('fail_count', 0)
                    info.success_count = token_data.get('success_count', 0)
                    info.average_response_time = token_data.get('average_response_time', 0)
                    info.total_requests = token_data.get('total_requests', 0)

                    # 恢复状态枚举
                    status_str = token_data.get('status', 'unknown')
                    try:
                        info.status = TokenStatus(status_str)
                    except ValueError:
                        info.status = TokenStatus.UNKNOWN

            print(f"[AccountPool] Loaded {len(data.get('tokens', []))} tokens from file")
        except Exception as e:
            print(f"[AccountPool] Failed to load from file: {e}")

    async def _login_all_accounts(self):
        """登录所有账号获取Token"""
        if not self.accounts:
            return

        print(f"[AccountPool] Logging in {len(self.accounts)} accounts...")

        for email, account in self.accounts.items():
            if not account.token or self._is_token_expired(account.token):
                try:
                    token = await self._login_account(email, account.password)
                    if token:
                        account.token = token
                        account.last_login = datetime.now().isoformat()
                        account.login_error = None

                        if token not in self.tokens:
                            self.tokens[token] = TokenInfo(token=token)
                        print(f"[AccountPool] Login successful: {email}")
                except Exception as e:
                    account.login_error = str(e)
                    print(f"[AccountPool] Login failed for {email}: {e}")

    async def _login_account(self, email: str, password: str) -> Optional[str]:
        """登录单个账号获取Token (使用 httpx 和代理管理器)"""
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
            "x-app-version": "20241129.1",
            "x-client-locale": "zh_CN",
            "x-client-platform": "web",
            "x-client-timezone-offset": "28800",
            "x-client-version": "1.8.0",
            "referrer": "https://chat本回答由 AI 生成，内容仅供参考，请仔细甄别。INCOMPLETE