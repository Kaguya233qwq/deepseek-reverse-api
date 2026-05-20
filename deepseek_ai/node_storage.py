"""节点存储模块"""

import json
import os
import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """节点信息"""

    address: str
    port: int
    name: str = ""
    protocol: str = "vless"
    latency: Optional[float] = None
    is_available: bool = True
    last_check: Optional[float] = None
    fail_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "address": self.address,
            "port": self.port,
            "name": self.name,
            "protocol": self.protocol,
            "latency": self.latency,
            "is_available": self.is_available,
            "last_check": self.last_check,
            "fail_count": self.fail_count,
        }


class NodeStorage:
    """节点存储"""

    def __init__(self, storage_path: str = "vless_nodes.json"):
        self.storage_path = storage_path
        self.nodes: List[NodeInfo] = []
        self._load()

    def _load(self):
        """从文件加载"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.nodes = [NodeInfo(**n) for n in data.get("nodes", [])]
                logger.info(f"[NodeStorage] 加载了 {len(self.nodes)} 个节点")
        except Exception as e:
            logger.warning(f"[NodeStorage] 加载失败: {e}")

    def save(self):
        """保存到文件"""
        try:
            data = {"saved_at": time.time(), "nodes": [n.to_dict() for n in self.nodes]}

            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"[NodeStorage] 已保存到 {self.storage_path}")
        except Exception as e:
            logger.error(f"[NodeStorage] 保存失败: {e}")

    def add_node(self, node: NodeInfo) -> bool:
        """添加节点"""
        # 检查是否已存在
        for n in self.nodes:
            if n.address == node.address and n.port == node.port:
                return False

        self.nodes.append(node)
        self.save()
        return True

    def update_node(self, address: str, port: int, **kwargs):
        """更新节点"""
        for node in self.nodes:
            if node.address == address and node.port == port:
                for key, value in kwargs.items():
                    if hasattr(node, key):
                        setattr(node, key, value)
                self.save()
                return True
        return False

    def get_available_nodes(self) -> List[NodeInfo]:
        """获取可用节点"""
        return [n for n in self.nodes if n.is_available]

    def get_stats(self) -> Dict:
        """获取统计"""
        total = len(self.nodes)
        available = sum(1 for n in self.nodes if n.is_available)

        return {
            "total": total,
            "available": available,
            "unavailable": total - available,
            "nodes": [n.to_dict() for n in self.nodes],
        }


# 全局存储
_node_storage: Optional[NodeStorage] = None


def get_node_storage() -> NodeStorage:
    """获取全局存储"""
    global _node_storage
    if _node_storage is None:
        _node_storage = NodeStorage()
    return _node_storage


def init_node_storage(storage_path: str = "vless_nodes.json") -> NodeStorage:
    """初始化存储"""
    global _node_storage
    if _node_storage is None:
        _node_storage = NodeStorage(storage_path)
    return _node_storage
