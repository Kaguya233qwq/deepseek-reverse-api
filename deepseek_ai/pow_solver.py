"""DeepSeek POW Solver - Direct Python WASM implementation"""

from __future__ import annotations

import json
import base64
import struct
import os
import logging
import asyncio
import wasmtime
from typing import Optional, TypedDict
from pathlib import Path

logger = logging.getLogger(__name__)


class ChallengeDict(TypedDict, total=False):
    algorithm: str
    challenge: str
    salt: str
    difficulty: int | float
    expire_at: int
    signature: str


class DeepSeekHashWasmtime:
    """DeepSeek Hash using wasmtime - Exact port from Node.js"""

    def __init__(self, wasm_path: str):
        self.wasm_path = wasm_path
        self.engine = wasmtime.Engine()
        self.store = wasmtime.Store(self.engine)

        with open(wasm_path, "rb") as f:
            wasm_bytes = f.read()

        module = wasmtime.Module(self.engine, wasm_bytes)
        self.instance = wasmtime.Instance(self.store, module, [])

        # Get exports
        self.memory = self.instance.exports(self.store)["memory"]
        self.add_to_stack_pointer = self.instance.exports(self.store)[
            "__wbindgen_add_to_stack_pointer"
        ]
        self.export_0 = self.instance.exports(self.store)["__wbindgen_export_0"]
        self.export_1 = self.instance.exports(self.store)["__wbindgen_export_1"]
        self.wasm_solve = self.instance.exports(self.store)["wasm_solve"]

        self.offset = 0

    def _get_memory_view(self):
        """Get memory as bytes-like object"""
        data_ptr = self.memory.data_ptr(self.store)  # type: ignore
        data_len = self.memory.data_len(self.store)  # type: ignore
        import ctypes

        ptr_value = ctypes.cast(data_ptr, ctypes.c_void_p).value
        if ptr_value is None:
            raise RuntimeError("Failed to obtain memory pointer from WASM runtime.")
        return (ctypes.c_uint8 * data_len).from_address(ptr_value)

    def _encode_string(self, text: str, allocate, reallocate=None):
        """Encode string to WASM memory - EXACT port from Node.js"""
        text_bytes = text.encode("utf-8")

        if reallocate is None:
            # Simple path
            ptr = allocate(self.store, len(text_bytes), 1)
            ptr = int(ptr) & 0xFFFFFFFF

            memory = self._get_memory_view()
            for i, byte in enumerate(text_bytes):
                memory[ptr + i] = byte

            self.offset = len(text_bytes)
            return ptr

        # Complex path with reallocate
        str_length = len(text)
        ptr = allocate(self.store, str_length, 1)
        ptr = int(ptr) & 0xFFFFFFFF

        memory = self._get_memory_view()
        ascii_length = 0

        # Write ASCII characters
        for i in range(str_length):
            char_code = ord(text[i])
            if char_code > 127:
                break
            memory[ptr + i] = char_code
            ascii_length += 1

        if ascii_length != str_length:
            # Handle non-ASCII
            if ascii_length > 0:
                text = text[ascii_length:]

            text_bytes = text.encode("utf-8")
            new_size = ascii_length + len(text_bytes)
            ptr = reallocate(self.store, ptr, str_length, new_size, 1)
            ptr = int(ptr) & 0xFFFFFFFF

            # Re-get memory view after realloc
            memory = self._get_memory_view()

            # Write remaining bytes
            for i, byte in enumerate(text_bytes):
                memory[ptr + ascii_length + i] = byte

            ascii_length += len(text_bytes)

        self.offset = ascii_length
        return ptr

    def calculate_hash(
        self,
        algorithm: str,
        challenge: str,
        salt: str,
        difficulty: int | float,
        expire_at: int,
    ):
        """Calculate hash answer"""
        if algorithm != "DeepSeekHashV1":
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        prefix = f"{salt}_{expire_at}_"

        try:
            # Allocate stack space
            retptr = self.add_to_stack_pointer(self.store, -16)  # type: ignore
            retptr = int(retptr) & 0xFFFFFFFF

            # Encode challenge
            ptr0 = self._encode_string(challenge, self.export_0, self.export_1)
            len0 = self.offset

            # Encode prefix
            ptr1 = self._encode_string(prefix, self.export_0, self.export_1)
            len1 = self.offset

            # Call wasm_solve
            self.wasm_solve(
                self.store, retptr, ptr0, len0, ptr1, len1, float(difficulty)
            )  # type: ignore

            # Read result
            memory = self._get_memory_view()

            # Read status (Int32, little-endian)
            status = int.from_bytes(
                bytes(memory[retptr + i] for i in range(4)),
                byteorder="little",
                signed=True,
            )

            # Read value (Float64, little-endian)
            value_bytes = bytes(memory[retptr + 8 + i] for i in range(8))
            value = struct.unpack("<d", value_bytes)[0]

            if status == 0:
                return None

            return int(value)

        finally:
            self.add_to_stack_pointer(self.store, 16)  # type: ignore


# Global instance
_hash_instance = None
_wasm_path: Optional[str] = None
_init_lock = asyncio.Lock()


def _find_wasm_file() -> str:
    """Find WASM file robustly using pathlib."""
    global _wasm_path
    if _wasm_path is not None:
        return _wasm_path

    base_dir = Path(__file__).resolve().parent
    wasm_name = "sha3_wasm_bg.7b9ca65ddd.wasm"

    possible_paths: list[Path] = [
        base_dir / wasm_name,
        base_dir.parent / wasm_name,
    ]

    if env_path := os.environ.get("DEEPSEEK_WASM_PATH"):
        possible_paths.insert(0, Path(env_path))

    for path in possible_paths:
        if path.is_file():
            resolved_path = str(path.resolve())
            logger.info(f"Discovered WASM payload at: {resolved_path}")
            _wasm_path = resolved_path
            return resolved_path

    logger.error("WASM payload module not found.")
    raise FileNotFoundError(
        "WASM file not found. Ensure the wasm file is present or DEEPSEEK_WASM_PATH is set."
    )


def get_deepseek_hash() -> DeepSeekHashWasmtime:
    """Get DeepSeekHash singleton instance (Sync path - assumes pre-initialized or safe to block briefly)"""
    global _hash_instance
    if _hash_instance is not None:
        return _hash_instance

    wasm_path = _find_wasm_file()

    try:
        _hash_instance = DeepSeekHashWasmtime(wasm_path)
        logger.info("Initialized POW solver via Wasmtime runtime.")
        return _hash_instance
    except Exception as e:
        logger.error(f"Wasmtime init failed: {e}")
        raise RuntimeError(f"Failed to initialize Wasmtime runtime: {e}")


async def get_deepseek_hash_async() -> DeepSeekHashWasmtime:
    """Get DeepSeekHash singleton instance (Async safe)"""
    global _hash_instance
    if _hash_instance is not None:
        return _hash_instance

    async with _init_lock:
        if _hash_instance is not None:
            return _hash_instance

        wasm_path = _find_wasm_file()

        try:
            _hash_instance = DeepSeekHashWasmtime(wasm_path)
            logger.info("Initialized POW solver via Wasmtime runtime.")
            return _hash_instance
        except Exception as e:
            logger.error(f"Wasmtime init failed: {e}")
            raise RuntimeError(f"Failed to initialize Wasmtime runtime: {e}")


def calculate_challenge_answer(challenge: dict | ChallengeDict) -> str:
    """Calculate challenge answer and return base64 encoded string"""
    algorithm = challenge.get("algorithm")
    challenge_str = challenge.get("challenge")
    salt = challenge.get("salt")
    difficulty = challenge.get("difficulty", 0)
    expire_at = challenge.get("expire_at", 0)
    signature = challenge.get("signature")

    if not all([algorithm, challenge_str, salt, difficulty, expire_at]):
        raise ValueError(
            f"Incomplete challenge dict. Missing required fields: {challenge}"
        )

    hash_calculator = get_deepseek_hash()
    answer = hash_calculator.calculate_hash(
        str(algorithm),
        str(challenge_str),
        str(salt),
        difficulty,
        int(expire_at),  # type: ignore
    )

    if answer is None:
        raise ValueError("Challenge calculation failed - WASM returned no answer")

    challenge_answer = {
        "algorithm": algorithm,
        "challenge": challenge_str,
        "salt": salt,
        "answer": answer,
        "signature": signature,
        "target_path": "/api/v0/chat/completion",
    }

    return base64.b64encode(
        json.dumps(challenge_answer, separators=(",", ":")).encode()
    ).decode()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mock_challenge: ChallengeDict = {
        "algorithm": "DeepSeekHashV1",
        "challenge": "0123456789abcdef" * 2,
        "salt": "mock_salt_test_123",
        "difficulty": 10,
        "expire_at": 1716120000,
        "signature": "mock_signature_xyz",
    }

    try:
        print("Testing WASM initialization and POW calculation with Wasmtime...")
        answer = calculate_challenge_answer(mock_challenge)
        print(f"Success! Calculated Hash Answer Base64:\n{answer}")
    except Exception as e:
        print(f"Failed during POW test: {e}")
