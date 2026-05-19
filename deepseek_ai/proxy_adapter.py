"""代理适配器 - 支持 VLess 代理（基于 httpx）"""

import threading
import os
from typing import Optional, Dict, Any

import httpx

from .vless_proxy import VlessProxyPool, get_proxy_pool, init_proxy_pool_from_env
from .vless_transport import VlessTransport, AsyncVlessTransport


class ProxyManager:
    """代理管理器 - 统一管理各种代理（基于 httpx）"""

    def __init__(self):
        self.vless_pool: Optional[VlessProxyPool] = None
        self.http_proxy: Optional[str] = None
        self.https_proxy: Optional[str] = None
        self._initialized = False

    def init_from_env(self) -> "ProxyManager":
        """从环境变量初始化"""
        # 初始化 VLess 代理池
        self.vless_pool = init_proxy_pool_from_env()

        # 读取 HTTP 代理设置
        self.http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        self.https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get(
            "https_proxy"
        )

        self._initialized = True
        return self

    def init_vless_from_file(self, filepath: str) -> "ProxyManager":
        """从文件加载 VLess 代理"""
        if self.vless_pool is None:
            self.vless_pool = get_proxy_pool()
        self.vless_pool.add_proxies_from_file(filepath)
        return self

    def add_vless_proxy(self, uri: str) -> bool:
        """添加单个 VLess 代理"""
        if self.vless_pool is None:
            self.vless_pool = get_proxy_pool()
        return self.vless_pool.add_proxy(uri)

    def get_requests_proxies(self) -> Optional[Dict[str, str]]:
        """获取 HTTP 代理配置（用于非 VLess 代理）"""
        proxies = {}

        if self.http_proxy:
            proxies["http://"] = self.http_proxy
        if self.https_proxy:
            proxies["https://"] = self.https_proxy

        return proxies if proxies else None

    def create_client(
        self, use_vless: bool = True, async_mode: bool = False, timeout: float = 30.0
    ) -> httpx.Client:
        """
        创建配置了代理的 httpx Client

        Args:
            use_vless: 是否使用 VLess 代理
            async_mode: 是否使用异步模式
            timeout: 超时时间

        Returns:
            配置好的 Client
        """
        if not self._initialized:
            self.init_from_env()

        if use_vless and self.vless_pool and self.vless_pool.count > 0:
            # 使用 VLess 代理
            if async_mode:
                transport = AsyncVlessTransport(proxy_pool=self.vless_pool)
                return httpx.AsyncClient(transport=transport, timeout=timeout)
            else:
                transport = VlessTransport(proxy_pool=self.vless_pool)
                return httpx.Client(transport=transport, timeout=timeout)
        else:
            # 使用普通 HTTP 代理
            proxies = self.get_requests_proxies()
            if async_mode:
                return httpx.AsyncClient(proxy=proxies, timeout=timeout)
            else:
                return httpx.Client(proxy=proxies, timeout=timeout)

    def create_session(self, use_vless: bool = True) -> httpx.Client:
        """
        创建配置了代理的 Session（兼容旧接口，返回同步 Client）

        Args:
            use_vless: 是否使用 VLess 代理

        Returns:
            配置好的 Client
        """
        return self.create_client(use_vless=use_vless, async_mode=False)

    def create_async_session(self, use_vless: bool = True) -> httpx.AsyncClient:
        """
        创建异步 Client

        Args:
            use_vless: 是否使用 VLess 代理

        Returns:
            配置好的 AsyncClient
        """
        return self.create_client(use_vless=use_vless, async_mode=True)

    def get_stats(self) -> Dict[str, Any]:
        """获取代理统计信息"""
        stats = {
            "http_proxy": self.http_proxy,
            "https_proxy": self.https_proxy,
        }

        if self.vless_pool:
            stats["vless"] = self.vless_pool.get_stats()
        else:
            stats["vless"] = {"total": 0, "healthy": 0, "unhealthy": 0, "proxies": []}

        return stats


# 全局代理管理器
_global_proxy_manager: Optional[ProxyManager] = None
_proxy_manager_lock = threading.Lock()


def get_proxy_manager() -> ProxyManager:
    """获取全局代理管理器（线程安全）"""
    global _global_proxy_manager
    if _global_proxy_manager is None:
        with _proxy_manager_lock:
            if _global_proxy_manager is None:
                _global_proxy_manager = ProxyManager()
    return _global_proxy_manager


def init_proxy_manager() -> ProxyManager:
    """初始化全局代理管理器（从环境变量，线程安全）"""
    global _global_proxy_manager
    with _proxy_manager_lock:
        if _global_proxy_manager is None:
            _global_proxy_manager = ProxyManager()
        if not _global_proxy_manager._initialized:
            _global_proxy_manager.init_from_env()
    return _global_proxy_manager
