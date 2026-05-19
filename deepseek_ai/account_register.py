"""DeepSeek AI 账号注册和登录工具

支持手动登录、自动登录和批量登录
"""

import json
import time
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://chat.deepseek.com"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


@dataclass
class RegistrationResult:
    """注册/登录结果"""
    success: bool
    email: str
    token: Optional[str] = None
    error: Optional[str] = None
    proxy: Optional[str] = None


class DeepSeekAccountRegister:
    """DeepSeek 账号注册和登录 (httpx based)"""

    def __init__(self, proxy: Optional[str] = None):
        """
        初始化

        Args:
            proxy: 代理地址，如 http://proxy:port
        """
        self.proxy = proxy
        # 使用 httpx.Client 替代 requests.Session
        self.client = httpx.Client(timeout=30.0)

        if proxy:
            self.client.proxies = {
                'http://': proxy,
                'https://': proxy
            }

    def signin(self, email: str, password: str) -> str:
        """
        登录获取 Token

        Args:
            email: 邮箱
            password: 密码

        Returns:
            Token

        Raises:
            Exception: 登录失败
        """
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
            "referrer": "https://chat.deepseek.com/sign_in",
            "User-Agent": USER_AGENT
        }

        data = {
            "email": email,
            "mobile": "",
            "password": password,
            "area_code": "",
            "device_id": "",
            "os": "web"
        }

        response = self.client.post(
            f"{DEEPSEEK_BASE_URL}/api/v0/users/login",
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            raise Exception(f"登录失败: HTTP {response.status_code}, {response.text}")

        result = response.json()

        if result.get("code") != 0:
            error_msg = result.get("msg") or result.get("data", {}).get("biz_msg") or "Unknown error"
            raise Exception(f"登录失败: {error_msg}")

        # 提取 Token
        biz_data = result.get("data", {}).get("biz_data", {})
        user = biz_data.get("user", {})
        token = user.get("token")

        if not token:
            raise Exception("登录成功但未获取到 Token")

        return token

    def close(self):
        """关闭 httpx client"""
        self.client.close()


def register_account_auto(
        email: str,
        password: str,
        proxy: Optional[str] = None,
        callback: Optional[Callable] = None
) -> RegistrationResult:
    """
    自动登录获取 Token

    Args:
        email: 邮箱
        password: 密码
        proxy: 代理地址
        callback: 回调函数

    Returns:
        注册结果
    """
    register = None
    try:
        logger.info(f"[Register] 正在登录: {email}")

        register = DeepSeekAccountRegister(proxy=proxy)
        token = register.signin(email, password)

        logger.info(f"[Register] 登录成功: {email}")

        if callback:
            callback(f"登录成功: {email}")

        return RegistrationResult(
            success=True,
            email=email,
            token=token,
            proxy=proxy
        )

    except Exception as e:
        logger.error(f"[Register] 登录失败 {email}: {e}")

        if callback:
            callback(f"登录失败: {email} - {e}")

        return RegistrationResult(
            success=False,
            email=email,
            error=str(e),
            proxy=proxy
        )
    finally:
        if register:
            register.close()
