"""节点测试模块"""

import time
import logging
import socket
from typing import Dict, Optional
from dataclasses import dataclass

from .node_storage import NodeStorage, get_node_storage

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """测试结果"""

    address: str
    port: int
    success: bool
    latency: Optional[float] = None
    error: Optional[str] = None


class NodeTester:
    """节点测试器"""

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def test_node(self, address: str, port: int) -> TestResult:
        """测试单个节点"""
        start_time = time.time()

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)

            result = sock.connect_ex((address, port))
            sock.close()

            latency = time.time() - start_time

            if result == 0:
                return TestResult(
                    address=address, port=port, success=True, latency=latency
                )
            else:
                return TestResult(
                    address=address,
                    port=port,
                    success=False,
                    error=f"Connection failed with code {result}",
                )

        except Exception as e:
            return TestResult(address=address, port=port, success=False, error=str(e))

    def test_all_nodes(
        self, storage: Optional[NodeStorage] = None
    ) -> Dict[str, TestResult]:
        """测试所有节点"""
        if storage is None:
            storage = get_node_storage()

        results = {}

        for node in storage.nodes:
            result = self.test_node(node.address, node.port)
            results[f"{node.address}:{node.port}"] = result

            # 更新节点状态
            storage.update_node(
                node.address,
                node.port,
                is_available=result.success,
                latency=result.latency,
                last_check=time.time(),
            )

        return results


# 全局测试器
_node_tester: Optional[NodeTester] = None


def get_node_tester() -> NodeTester:
    """获取全局测试器"""
    global _node_tester
    if _node_tester is None:
        _node_tester = NodeTester()
    return _node_tester


def init_node_tester(timeout: float = 5.0) -> NodeTester:
    """初始化测试器"""
    global _node_tester
    if _node_tester is None:
        _node_tester = NodeTester(timeout)
    return _node_tester
