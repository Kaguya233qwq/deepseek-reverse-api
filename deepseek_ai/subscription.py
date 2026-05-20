"""订阅管理模块 - 管理 Vless 节点订阅"""

import base64
import logging
import os
import time
import urllib.parse
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)


@dataclass
class VlessNode:
    """Vless 节点信息"""

    name: str
    address: str
    port: int
    uuid: str
    security: str = "tls"
    network: str = "ws"
    host: str = ""
    path: str = "/"
    sni: str = ""
    raw_url: str = ""

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "address": self.address,
            "port": self.port,
            "uuid": self.uuid,
            "security": self.security,
            "network": self.network,
            "host": self.host,
            "path": self.path,
            "sni": self.sni,
            "raw_url": self.raw_url,
        }


@dataclass
class Subscription:
    """订阅信息"""

    url: str
    nodes: List[VlessNode] = field(default_factory=list)
    last_update: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "url": self.url,
            "nodes": [n.to_dict() for n in self.nodes],
            "last_update": self.last_update,
        }


class SubscriptionManager:
    """订阅管理器"""

    def __init__(self):
        self.subscriptions: List[Subscription] = []
        self._nodes_cache: List[VlessNode] = []
        self._pattern: str = ""

    def add_subscription(self, url: str) -> bool:
        """添加订阅"""
        try:
            sub = Subscription(url=url)
            self.subscriptions.append(sub)
            logger.info(f"[Subscription] 添加订阅: {url}")
            return True
        except Exception as e:
            logger.error(f"[Subscription] 添加订阅失败: {e}")
            return False

    def refresh_all(self) -> Dict[str, int]:
        """刷新所有订阅"""
        results = {}
        for sub in self.subscriptions:
            try:
                count = self._fetch_subscription(sub)
                results[sub.url] = count
            except Exception as e:
                logger.error(f"[Subscription] 刷新失败 {sub.url}: {e}")
                results[sub.url] = 0

        # 更新缓存
        self._update_cache()
        return results

    def _fetch_subscription(self, sub: Subscription) -> int:
        """获取订阅内容"""
        try:
            resp = requests.get(sub.url, timeout=30)
            resp.raise_for_status()

            # 解码 base64
            content = base64.b64decode(resp.text).decode("utf-8")

            # 解析节点
            nodes = self._parse_nodes(content)
            sub.nodes = nodes
            sub.last_update = time.time()

            logger.info(f"[Subscription] 获取到 {len(nodes)} 个节点")
            return len(nodes)

        except Exception as e:
            logger.error(f"[Subscription] 获取订阅失败: {e}")
            return 0

    def _parse_nodes(self, content: str) -> List[VlessNode]:
        """解析节点"""
        nodes = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("vless://"):
                continue

            try:
                node = self._parse_vless_url(line)
                if node:
                    nodes.append(node)
            except Exception as e:
                logger.warning(f"[Subscription] 解析节点失败: {e}")

        return nodes

    def _parse_vless_url(self, url: str) -> Optional[VlessNode]:
        """解析 Vless URL"""
        # vless://uuid@address:port?params#name
        try:
            # 移除 vless:// 前缀
            content = url[8:]

            # 分离备注
            if "#" in content:
                content, name_encoded = content.split("#", 1)
                name = urllib.parse.unquote(name_encoded)
            else:
                name = "Unknown"

            # 分离参数
            if "?" in content:
                main_part, params_part = content.split("?", 1)
            else:
                main_part = content
                params_part = ""

            # 解析主体: uuid@address:port
            uuid, server_part = main_part.split("@", 1)
            address, port_str = server_part.rsplit(":", 1)
            port = int(port_str)

            # 解析参数
            params = urllib.parse.parse_qs(params_part)

            return VlessNode(
                name=name,
                address=address,
                port=port,
                uuid=uuid,
                security=params.get("security", ["tls"])[0],
                network=params.get("type", ["ws"])[0],
                host=params.get("host", [""])[0],
                path=params.get("path", ["/"])[0],
                sni=params.get("sni", [""])[0],
                raw_url=url,
            )

        except Exception as e:
            logger.warning(f"[Subscription] 解析 Vless URL 失败: {e}")
            return None

    def _update_cache(self):
        """更新节点缓存"""
        self._nodes_cache = []
        for sub in self.subscriptions:
            self._nodes_cache.extend(sub.nodes)

    def set_pattern(self, pattern: str):
        """设置节点匹配规则"""
        self._pattern = pattern
        logger.info(f"[Subscription] 设置匹配规则: {pattern}")

    def get_filtered_nodes(self) -> List[VlessNode]:
        """获取筛选后的节点"""
        if not self._pattern:
            return self._nodes_cache

        filtered = []
        for node in self._nodes_cache:
            if self._pattern in node.name:
                filtered.append(node)

        return filtered

    def get_random_node(self) -> Optional[VlessNode]:
        """获取随机节点"""
        import random

        nodes = self.get_filtered_nodes()
        if not nodes:
            return None
        return random.choice(nodes)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        total = len(self._nodes_cache)
        filtered = len(self.get_filtered_nodes())

        return {
            "subscriptions": len(self.subscriptions),
            "total_nodes": total,
            "filtered_nodes": filtered,
            "pattern": self._pattern,
            "current_pattern": {"pattern": self._pattern, "available": filtered},
        }


# 全局订阅管理器
_subscription_manager: Optional[SubscriptionManager] = None


def get_subscription_manager() -> SubscriptionManager:
    """获取全局订阅管理器"""
    global _subscription_manager
    if _subscription_manager is None:
        _subscription_manager = SubscriptionManager()
    return _subscription_manager


def init_subscriptions_from_env() -> SubscriptionManager:
    """从环境变量初始化订阅"""
    global _subscription_manager

    if _subscription_manager is None:
        _subscription_manager = SubscriptionManager()

        # 从环境变量读取订阅URL
        urls_str = os.environ.get("VLESS_SUBSCRIPTION_URLS", "")
        if urls_str:
            for url in urls_str.split(","):
                url = url.strip()
                if url:
                    _subscription_manager.add_subscription(url)

        # 从环境变量读取匹配规则
        pattern = os.environ.get("VLESS_NODE_PATTERN", "")
        if pattern:
            _subscription_manager.set_pattern(pattern)

        # 刷新订阅
        _subscription_manager.refresh_all()

    return _subscription_manager
