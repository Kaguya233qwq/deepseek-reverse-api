"""VLess Transport Layer - 支持同步和异步传输

提供 VLess 协议的传输层抽象，支持：
- TCP 传输（明文或 TLS）
- WebSocket 传输（ws 或 wss）
- 同步和异步两种接口
"""

import socket
import ssl
import struct
import base64
import os
import asyncio
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VLessProtocolError(Exception):
    """VLess 协议错误"""

    pass


class VLessHandshakeError(VLessProtocolError):
    """握手错误"""

    pass


class VLessTransportError(VLessProtocolError):
    """传输错误"""

    pass


class VLessRequestHeader:
    """VLess 请求头构建器"""

    VERSION = 0
    COMMAND_TCP = 1
    COMMAND_UDP = 2
    COMMAND_MUX = 3

    ADDR_TYPE_IPV4 = 1
    ADDR_TYPE_DOMAIN = 2
    ADDR_TYPE_IPV6 = 3

    @staticmethod
    def build(
        uuid: str, target_host: str, target_port: int, command: int = COMMAND_TCP
    ) -> bytes:
        """
        构建 VLess 请求头

        Args:
            uuid: UUID 字符串
            target_host: 目标主机
            target_port: 目标端口
            command: 命令类型 (TCP/UDP/MUX)

        Returns:
            请求头字节数据
        """
        # 验证并转换 UUID
        try:
            uuid_bytes = bytes.fromhex(uuid.replace("-", ""))
            if len(uuid_bytes) != 16:
                raise ValueError("Invalid UUID length")
        except Exception as e:
            raise VLessProtocolError(f"Invalid UUID format: {e}")

        header = bytearray()

        # Version (1 byte)
        header.append(VLessRequestHeader.VERSION)

        # UUID (16 bytes)
        header.extend(uuid_bytes)

        # Command (1 byte)
        header.append(command)

        # Address Type and Address
        try:
            # 尝试 IPv4
            socket.inet_pton(socket.AF_INET, target_host)
            header.append(VLessRequestHeader.ADDR_TYPE_IPV4)
            header.extend(socket.inet_pton(socket.AF_INET, target_host))
        except OSError:
            try:
                # 尝试 IPv6
                socket.inet_pton(socket.AF_INET6, target_host)
                header.append(VLessRequestHeader.ADDR_TYPE_IPV6)
                header.extend(socket.inet_pton(socket.AF_INET6, target_host))
            except OSError:
                # 域名
                domain_bytes = target_host.encode("utf-8")
                if len(domain_bytes) > 255:
                    raise VLessProtocolError("Domain name too long")
                header.append(VLessRequestHeader.ADDR_TYPE_DOMAIN)
                header.append(len(domain_bytes))
                header.extend(domain_bytes)

        # Port (2 bytes, big-endian)
        header.extend(struct.pack(">H", target_port))

        return bytes(header)


class VlessTransport:
    """同步 VLess 传输层"""

    def __init__(self, uri: str):
        """
        初始化同步 VLess 传输

        Args:
            uri: VLess URI (vless://...)
        """
        from deepseek_ai.vless_proxy import VlessURI

        self.config = VlessURI(uri)
        self._socket: Optional[socket.socket] = None
        self._ssl_socket: Optional[ssl.SSLSocket] = None

    def connect(
        self, target_host: str, target_port: int, timeout: float = 30
    ) -> socket.socket:
        """
        通过 VLess 代理连接到目标主机（返回原始 socket）

        Args:
            target_host: 目标主机
            target_port: 目标端口
            timeout: 超时时间（秒）

        Returns:
            已连接的 socket 对象
        """
        try:
            # 1. 连接到 VLess 服务器
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((self.config.address, self.config.port))

            # 2. 如果需要 TLS，升级连接
            if self.config.tls:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                server_hostname = self.config.sni or self.config.address
                ssl_sock = ssl_context.wrap_socket(
                    sock, server_hostname=server_hostname
                )
                self._ssl_socket = ssl_sock
                sock = ssl_sock

            # 3. 发送 VLess 请求头
            request_header = VLessRequestHeader.build(
                self.config.uuid, target_host, target_port
            )
            sock.sendall(request_header)

            # 4. 读取响应（VLess 协议成功时无响应，直接开始数据传输）
            # 某些实现可能返回一个字节的状态码，尝试非阻塞读取
            sock.settimeout(0.1)
            try:
                response = sock.recv(1)
                if response and response[0] != 0:
                    raise VLessHandshakeError(
                        f"Handshake failed with status: {response[0]}"
                    )
            except socket.timeout:
                # 没有响应是正常的
                pass
            finally:
                sock.settimeout(timeout)

            self._socket = sock
            return sock

        except socket.timeout:
            raise VLessTransportError(
                f"Connection to {self.config.address}:{self.config.port} timed out"
            )
        except Exception as e:
            raise VLessTransportError(f"Failed to establish VLess connection: {e}")

    def connect_tcp(
        self, target_host: str, target_port: int, timeout: float = 30
    ) -> socket.socket:
        """TCP 传输方式连接到目标主机"""
        return self.connect(target_host, target_port, timeout)

    def connect_websocket(
        self,
        target_host: str,
        target_port: int,
        path: str = "/",
        host: str | None = None,
        timeout: float = 30,
    ) -> socket.socket:
        """
        WebSocket 传输方式连接到目标主机（简化实现）

        注意：完整的 WebSocket 支持需要实现帧编码/解码，
        这里提供基础框架，建议使用 aiohttp/websockets 库实现完整的 WS 客户端

        Args:
            target_host: 目标主机
            target_port: 目标端口
            path: WebSocket 路径
            host: Host 头
            timeout: 超时时间
        """
        # WebSocket 升级请求
        key = base64.b64encode(os.urandom(16)).decode()
        host_header = host or target_host

        ws_request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host_header}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )

        sock = self.connect(target_host, target_port, timeout)
        sock.sendall(ws_request.encode())

        # 读取响应
        response = sock.recv(1024).decode()
        if "101" not in response:
            raise VLessHandshakeError(f"WebSocket upgrade failed: {response}")

        return sock

    def close(self):
        """关闭连接"""
        if self._ssl_socket:
            self._ssl_socket.close()
        elif self._socket:
            self._socket.close()
        self._socket = None
        self._ssl_socket = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncVlessTransport:
    """异步 VLess 传输层"""

    def __init__(self, uri: str):
        """
        初始化异步 VLess 传输

        Args:
            uri: VLess URI (vless://...)
        """
        from deepseek_ai.vless_proxy import VlessURI

        self.config = VlessURI(uri)
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._transport = None

    async def connect(
        self, target_host: str, target_port: int, timeout: float = 30
    ) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """
        通过 VLess 代理连接到目标主机

        Args:
            target_host: 目标主机
            target_port: 目标端口
            timeout: 超时时间（秒）

        Returns:
            (reader, writer) 元组
        """
        try:
            # 1. 连接到 VLess 服务器
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.config.address, self.config.port),
                timeout=timeout,
            )

            # 2. 如果需要 TLS，升级连接
            if self.config.tls:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                server_hostname = self.config.sni or self.config.address

                loop = asyncio.get_event_loop()
                transport = writer.transport
                protocol = writer.transport.get_protocol()

                ssl_transport = await loop.start_tls(
                    transport, protocol, ssl_context, server_hostname=server_hostname
                )

                # 更新 reader/writer
                self._reader = asyncio.StreamReader()
                self._reader.set_transport(ssl_transport)
                self._writer = asyncio.StreamWriter(
                    ssl_transport, protocol, self._reader, loop
                )
                reader, writer = self._reader, self._writer

            # 3. 发送 VLess 请求头
            request_header = VLessRequestHeader.build(
                self.config.uuid, target_host, target_port
            )
            writer.write(request_header)
            await writer.drain()

            # 4. 读取响应（可选）
            try:
                response = await asyncio.wait_for(reader.read(1), timeout=1)
                if response and response[0] != 0:
                    raise VLessHandshakeError(
                        f"Handshake failed with status: {response[0]}"
                    )
            except asyncio.TimeoutError:
                # 没有响应是正常的
                pass

            self._reader, self._writer = reader, writer
            return reader, writer

        except asyncio.TimeoutError:
            raise VLessTransportError(
                f"Connection to {self.config.address}:{self.config.port} timed out"
            )
        except Exception as e:
            raise VLessTransportError(
                f"Failed to establish async VLess connection: {e}"
            )

    async def connect_tcp(
        self, target_host: str, target_port: int, timeout: float = 30
    ) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """TCP 传输方式连接到目标主机"""
        return await self.connect(target_host, target_port, timeout)

    async def connect_websocket(
        self,
        target_host: str,
        target_port: int,
        path: str = "/",
        host: str = None,
        timeout: float = 30,
    ) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """
        WebSocket 传输方式连接到目标主机

        注意：完整 WebSocket 支持需要实现帧协议，
        建议使用 websockets 或 aiohttp 库

        Args:
            target_host: 目标主机
            target_port: 目标端口
            path: WebSocket 路径
            host: Host 头
            timeout: 超时时间
        """
        # WebSocket 升级请求
        key = base64.b64encode(os.urandom(16)).decode()
        host_header = host or target_host

        ws_request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host_header}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )

        reader, writer = await self.connect(target_host, target_port, timeout)
        writer.write(ws_request.encode())
        await writer.drain()

        # 读取响应
        response = await asyncio.wait_for(reader.read(1024), timeout=5)
        response_str = response.decode()

        if "101" not in response_str:
            raise VLessHandshakeError(f"WebSocket upgrade failed: {response_str}")

        return reader, writer

    async def close(self):
        """关闭连接"""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
            self._reader = None
            self._writer = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# 辅助函数
def create_vless_transport(uri: str) -> VlessTransport:
    """创建同步 VLess 传输实例"""
    return VlessTransport(uri)


def create_async_vless_transport(uri: str) -> AsyncVlessTransport:
    """创建异步 VLess 传输实例"""
    return AsyncVlessTransport(uri)
