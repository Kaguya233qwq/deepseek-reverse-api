"""Vless Proxy Client - 支持 v2ray 兼容的 Vless 协议

支持 Vless 协议的 TCP 和 WebSocket 传输方式
"""

import struct
import socket
import ssl
import asyncio
import random
import os
from typing import Optional, Dict, Any, Tuple, List
from urllib.parse import parse_qs
import logging

logger = logging.getLogger(__name__)


class VlessURI:
    """解析 Vless URI"""
    
    def __init__(self, uri: str):
        self.uri: str = uri
        self.uuid: str = ""
        self.address: Optional[str] = None
        self.port: Optional[int] = None
        self.security: str = 'none'
        self.network: str = 'tcp'
        self.host: Optional[str] = None
        self.path: Optional[str] = None
        self.tls: bool = False
        self.sni: Optional[str] = None
        self.alpn: Optional[str] = None
        self.fp: Optional[str] = None
        self.pbk: Optional[str] = None
        self.sid: Optional[str] = None
        self.spx: Optional[str] = None
        self._parse()
    
    def _parse(self) -> None:
        """解析 Vless URI"""
        try:
            # vless://uuid@address:port?params#remark
            if not self.uri.startswith('vless://'):
                raise ValueError('Invalid Vless URI format')
            
            # 移除 vless:// 前缀
            content = self.uri[8:]
            
            # 分离备注
            if '#' in content:
                content, _ = content.split('#', 1)
            
            # 分离参数
            if '?' in content:
                main_part, params_part = content.split('?', 1)
            else:
                main_part = content
                params_part = ''
            
            # 解析主体部分: uuid@address:port
            if '@' not in main_part:
                raise ValueError('Invalid Vless URI: missing @')
            
            uuid_part, server_part = main_part.split('@', 1)
            self.uuid = uuid_part
            
            # 解析服务器地址和端口
            if ':' not in server_part:
                raise ValueError('Invalid Vless URI: missing port')
            
            # 处理 IPv6 地址
            if server_part.startswith('['):
                end_idx = server_part.find(']')
                if end_idx == -1:
                    raise ValueError('Invalid Vless URI: invalid IPv6 address')
                self.address = server_part[1:end_idx]
                port_part = server_part[end_idx + 1:]
                if port_part.startswith(':'):
                    self.port = int(port_part[1:])
                else:
                    raise ValueError('Invalid Vless URI: missing port after IPv6')
            else:
                addr_part, port_part = server_part.rsplit(':', 1)
                self.address = addr_part
                self.port = int(port_part)
            
            # 解析参数
            if params_part:
                params = parse_qs(params_part)
                
                self.security = params.get('security', ['none'])[0]
                self.network = params.get('type', ['tcp'])[0]
                self.host = params.get('host', [None])[0]
                self.path = params.get('path', ['/'])[0]
                self.sni = params.get('sni', [None])[0]
                self.alpn = params.get('alpn', [None])[0]
                self.fp = params.get('fp', [None])[0]
                self.pbk = params.get('pbk', [None])[0]
                self.sid = params.get('sid', [None])[0]
                self.spx = params.get('spx', [None])[0]
                
                if self.security in ['tls', 'xtls', 'reality']:
                    self.tls = True
                
        except Exception as e:
            raise ValueError(f'Failed to parse Vless URI: {e}')
    
    def __repr__(self) -> str:
        return f"VlessURI({self.address}:{self.port}, network={self.network}, tls={self.tls})"


class VlessProxy:
    """Vless 代理客户端"""
    
    # Vless 协议常量
    VERSION = 0
    COMMAND_TCP = 1
    COMMAND_UDP = 2
    COMMAND_MUX = 3
    
    # 地址类型
    ADDR_TYPE_IPV4 = 1
    ADDR_TYPE_DOMAIN = 2
    ADDR_TYPE_IPV6 = 3
    
    def __init__(self, uri: str):
        """
        初始化 Vless 代理
        
        Args:
            uri: Vless URI，格式: vless://uuid@address:port?params#remark
        """
        self.config: VlessURI = VlessURI(uri)
        self._lock: asyncio.Lock = asyncio.Lock()
        self._last_used: float = 0
        self._fail_count: int = 0
        self._healthy: bool = True
    
    @property
    def is_healthy(self) -> bool:
        """检查代理是否健康"""
        return self._healthy and self._fail_count < 3
    
    @property
    def identifier(self) -> str:
        """获取代理标识符"""
        return f"{self.config.address}:{self.config.port}"
    
    def mark_success(self) -> None:
        """标记请求成功"""
        self._fail_count = 0
        self._healthy = True
        self._last_used = asyncio.get_event_loop().time()
    
    def mark_fail(self) -> None:
        """标记请求失败"""
        self._fail_count += 1
        if self._fail_count >= 3:
            self._healthy = False
    
    def _make_request_header(self, target_host: str, target_port: int) -> bytes:
        """
        构建 Vless 请求头
        
        协议格式:
        +------------------+------------------+--------------------------------+
        |      1 Byte      |     16 Bytes     |           M Bytes              |
        +------------------+------------------+--------------------------------+
        |      Version     |      UUID        |          Request Header        |
        +------------------+------------------+--------------------------------+
        
        Request Header:
        +------------------+------------------+---------------+------------------+
        |      1 Byte      |      1 Byte      |    1 Byte     |    S Bytes       |
        +------------------+------------------+---------------+------------------+
        |      Command     |   Address Type   |  Address      |     Port         |
        +------------------+------------------+---------------+------------------+
        """
        # 验证 UUID
        try:
            uuid_bytes = bytes.fromhex(self.config.uuid.replace('-', ''))
            if len(uuid_bytes) != 16:
                raise ValueError('Invalid UUID length')
        except Exception as e:
            raise ValueError(f'Invalid UUID format: {e}')
        
        # 构建请求头
        header = bytearray()
        
        # Version (1 byte)
        header.append(self.VERSION)
        
        # UUID (16 bytes)
        header.extend(uuid_bytes)
        
        # Command (1 byte) - TCP
        header.append(self.COMMAND_TCP)
        
        # Address Type and Address
        try:
            # 尝试作为 IPv4
            socket.inet_pton(socket.AF_INET, target_host)
            header.append(self.ADDR_TYPE_IPV4)
            header.extend(socket.inet_pton(socket.AF_INET, target_host))
        except OSError:
            try:
                # 尝试作为 IPv6
                socket.inet_pton(socket.AF_INET6, target_host)
                header.append(self.ADDR_TYPE_IPV6)
                header.extend(socket.inet_pton(socket.AF_INET6, target_host))
            except OSError:
                # 作为域名
                domain_bytes = target_host.encode('utf-8')
                if len(domain_bytes) > 255:
                    raise ValueError('Domain name too long')
                header.append(self.ADDR_TYPE_DOMAIN)
                header.append(len(domain_bytes))
                header.extend(domain_bytes)
        
        # Port (2 bytes, big-endian)
        header.extend(struct.pack('>H', target_port))
        
        return bytes(header)
    
    async def create_connection(self, target_host: str, target_port: int) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """
        通过 Vless 代理创建到目标主机的连接
        
        Args:
            target_host: 目标主机地址
            target_port: 目标主机端口
            
        Returns:
            (reader, writer) 元组
        """
        try:
            # 连接到 Vless 服务器
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.config.address, self.config.port),
                timeout=10
            )
            
            # 如果启用了 TLS，包装连接
            if self.config.tls:
                ssl_context = ssl.create_default_context()
                if self.config.sni:
                    ssl_context.server_hostname = self.config.sni
                
                # 创建 SSL 连接
                loop = asyncio.get_event_loop()
                transport = writer.transport
                protocol = transport.get_protocol()
                
                # 升级连接到 SSL
                ssl_transport = await loop.start_tls(
                    transport, protocol, ssl_context,
                    server_hostname=self.config.sni or self.config.address
                )
                
                # 获取新的 reader 和 writer
                reader = asyncio.StreamReader()
                reader.set_transport(ssl_transport)
                writer = asyncio.StreamWriter(ssl_transport, protocol, reader, loop)
            
            # 发送 Vless 请求头
            request_header = self._make_request_header(target_host, target_port)
            writer.write(request_header)
            await writer.drain()
            
            # 读取响应（Vless 协议响应为空或包含状态）
            # Vless 协议在成功时没有响应，直接开始传输数据
            
            return reader, writer
            
        except asyncio.TimeoutError:
            raise ConnectionError(f'Connection to Vless server {self.config.address}:{self.config.port} timed out')
        except Exception as e:
            raise ConnectionError(f'Failed to create Vless connection: {e}')
    
    async def test_connection(self, target_host: str = 'www.google.com', target_port: int = 443, timeout: int = 10) -> bool:
        """
        测试代理连接
        
        Args:
            target_host: 测试目标主机
            target_port: 测试目标端口
            timeout: 超时时间（秒）
            
        Returns:
            连接是否成功
        """
        try:
            reader, writer = await asyncio.wait_for(
                self.create_connection(target_host, target_port),
                timeout=timeout
            )
            
            # 发送一个简单的 HTTP 请求来验证连接
            http_request = f'HEAD / HTTP/1.1\r\nHost: {target_host}\r\nConnection: close\r\n\r\n'
            writer.write(http_request.encode())
            await writer.drain()
            
            # 尝试读取响应
            response = await asyncio.wait_for(reader.read(1024), timeout=5)
            
            writer.close()
            await writer.wait_closed()
            
            if response:
                self.mark_success()
                return True
            return False
            
        except Exception as e:
            logger.debug(f'Vless proxy test failed for {self.identifier}: {e}')
            self.mark_fail()
            return False


class VlessProxyPool:
    """Vless 代理池 - 管理多个 Vless 代理"""
    
    def __init__(self):
        self._proxies: List[VlessProxy] = []
        self._current_index: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()
    
    def add_proxy(self, uri: str) -> bool:
        """
        添加 Vless 代理到池
        
        Args:
            uri: Vless URI
            
        Returns:
            是否添加成功
        """
        try:
            proxy = VlessProxy(uri)
            self._proxies.append(proxy)
            logger.info(f'Added Vless proxy: {proxy.identifier}')
            return True
        except Exception as e:
            logger.error(f'Failed to add Vless proxy: {e}')
            return False
    
    def add_proxies_from_uris(self, uris: List[str]) -> Tuple[int, int]:
        """
        从多个 URI 添加代理
        
        Args:
            uris: Vless URI 列表
            
        Returns:
            (成功数量, 失败数量)
        """
        success = 0
        failed = 0
        for uri in uris:
            if self.add_proxy(uri.strip()):
                success += 1
            else:
                failed += 1
        return success, failed
    
    def add_proxies_from_env(self, env_var: str = 'VLESS_PROXIES') -> Tuple[int, int]:
        """
        从环境变量添加代理
        
        Args:
            env_var: 环境变量名
            
        Returns:
            (成功数量, 失败数量)
        """
        uris_str = os.environ.get(env_var, '')
        if not uris_str:
            return 0, 0
        
        # 支持多种分隔符: 换行、逗号、分号
        uris = []
        for separator in ['\n', ',', ';']:
            if separator in uris_str:
                uris = [u.strip() for u in uris_str.split(separator) if u.strip()]
                break
        
        if not uris:
            uris = [uris_str.strip()]
        
        return self.add_proxies_from_uris(uris)
    
    def add_proxies_from_file(self, filepath: str) -> Tuple[int, int]:
        """
        从文件添加代理
        
        Args:
            filepath: 文件路径，每行一个 Vless URI
            
        Returns:
            (成功数量, 失败数量)
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                uris = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            return self.add_proxies_from_uris(uris)
        except Exception as e:
            logger.error(f'Failed to read proxy file {filepath}: {e}')
            return 0, 0
    
    @property
    def count(self) -> int:
        """获取代理总数"""
        return len(self._proxies)
    
    @property
    def healthy_count(self) -> int:
        """获取健康代理数量"""
        return sum(1 for p in self._proxies if p.is_healthy)
    
    def get_proxy(self, strategy: str = 'round_robin') -> Optional[VlessProxy]:
        """
        获取一个代理
        
        Args:
            strategy: 选择策略 ('round_robin' 或 'random')
            
        Returns:
            VlessProxy 实例或 None
        """
        if not self._proxies:
            return None
        
        # 过滤健康代理
        healthy_proxies = [p for p in self._proxies if p.is_healthy]
        if not healthy_proxies:
            # 如果没有健康代理，尝试使用所有代理
            healthy_proxies = self._proxies
        
        if strategy == 'random':
            return random.choice(healthy_proxies)
        else:  # round_robin
            with self._lock:
                proxy = healthy_proxies[self._current_index % len(healthy_proxies)]
                self._current_index += 1
                return proxy
    
    async def test_all_proxies(self, target_host: str = 'www.google.com', target_port: int = 443) -> Dict[str, bool]:
        """
        测试所有代理
        
        Args:
            target_host: 测试目标主机
            target_port: 测试目标端口
            
        Returns:
            代理标识符到测试结果的映射
        """
        results: Dict[str, bool] = {}
        tasks: List[Tuple[str, asyncio.Task]] = []
        
        for proxy in self._proxies:
            task = asyncio.create_task(proxy.test_connection(target_host, target_port))
            tasks.append((proxy.identifier, task))
        
        for identifier, task in tasks:
            try:
                result = await task
                results[identifier] = result
            except Exception as e:
                logger.error(f'Proxy test error for {identifier}: {e}')
                results[identifier] = False
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取代理池统计信息"""
        return {
            'total': self.count,
            'healthy': self.healthy_count,
            'unhealthy': self.count - self.healthy_count,
            'proxies': [
                {
                    'identifier': p.identifier,
                    'healthy': p.is_healthy,
                    'fail_count': p._fail_count,
                    'network': p.config.network,
                    'tls': p.config.tls
                }
                for p in self._proxies
            ]
        }


# 全局代理池实例
_global_proxy_pool: Optional[VlessProxyPool] = None
_proxy_pool_initialized: bool = False


def get_proxy_pool() -> VlessProxyPool:
    """获取全局代理池实例"""
    global _global_proxy_pool
    if _global_proxy_pool is None:
        _global_proxy_pool = VlessProxyPool()
    return _global_proxy_pool


def init_proxy_pool_from_env() -> VlessProxyPool:
    """从环境变量初始化代理池（只初始化一次）"""
    global _proxy_pool_initialized
    pool = get_proxy_pool()
    
    # 避免重复初始化
    if _proxy_pool_initialized:
        return pool
    
    pool.add_proxies_from_env('VLESS_PROXIES')
    
    # 也检查 VLESS_PROXY_FILE 环境变量
    proxy_file = os.environ.get('VLESS_PROXY_FILE')
    if proxy_file:
        pool.add_proxies_from_file(proxy_file)
    
    _proxy_pool_initialized = True
    return pool
