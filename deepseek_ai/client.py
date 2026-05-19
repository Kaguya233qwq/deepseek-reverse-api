"""DeepSeek Client - OpenAI compatible interface"""

from typing import List, Dict, Optional, Any, Generator, Union
from .adapter import DeepSeekAdapter
from .stream_handler import DeepSeekStreamHandler


class DeepSeekClient:
    """DeepSeek Client with OpenAI compatible interface"""

    def __init__(self, token: str, use_proxy: bool = True):
        """Initialize DeepSeek Client

        Args:
            token: DeepSeek token from login response
            use_proxy: Whether to use proxy (Vless or HTTP proxy)
        """
        self.adapter = DeepSeekAdapter(token, use_proxy=use_proxy)
        self._session_id: Optional[str] = None

    def chat_completions(
        self,
        model: str,
        messages: List[Dict],
        stream: bool = False,
        temperature: Optional[float] = None,
        web_search: bool = False,
        reasoning_effort: Optional[str] = None,
        thinking_enabled: Optional[bool] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Any] = None,
        auto_delete_session: bool = False,
    ) -> Union[Generator[str, None, None], Dict]:
        """Chat completions API

        Args:
            model: Model name
            messages: List of messages
            stream: Whether to use streaming
            temperature: Temperature setting
            web_search: Whether to enable web search
            reasoning_effort: Reasoning effort level ('low', 'medium', 'high')
            thinking_enabled: Whether to enable thinking mode (overrides reasoning_effort)
            tools: List of tools for function calling
            tool_choice: Tool choice configuration
            auto_delete_session: Whether to delete the session after completion

        Returns:
            Generator for streaming, dict for non-streaming
        """
        # Process messages with tool support
        processed_messages = messages.copy()

        if tools:
            has_tool_prompt = any(
                msg.get("role") == "system"
                and (
                    "Available Tools" in msg.get("content", "")
                    or "<tools>" in msg.get("content", "")
                )
                for msg in messages
            )

            if not has_tool_prompt:
                from .tool_parser import ToolParser

                tool_prompt = ToolParser.tools_to_system_prompt(tools)

                system_messages = [
                    msg for msg in processed_messages if msg.get("role") == "system"
                ]
                if system_messages:
                    system_messages[0]["content"] = (
                        system_messages[0]["content"] + "\n\n" + tool_prompt
                    )
                else:
                    processed_messages.insert(
                        0, {"role": "system", "content": tool_prompt}
                    )

        response, session_id = self.adapter.chat_completion(  # pyright: ignore[reportGeneralTypeIssues]
            model=model,
            messages=processed_messages,
            stream=True,  # Always stream from backend
            temperature=temperature,
            web_search=web_search,
            reasoning_effort=reasoning_effort,
            thinking_enabled=thinking_enabled,
        )

        self._session_id = session_id

        handler = DeepSeekStreamHandler(
            model,
            session_id,
            on_end=lambda: self._on_stream_end(auto_delete_session),
            web_search_enabled=web_search,
            reasoning_effort=reasoning_effort,
        )

        if stream:
            return handler.handle_stream(response)
        else:
            return handler.handle_non_stream(response)

    def _on_stream_end(self, auto_delete_session: bool):
        """Handle stream end callback"""
        if auto_delete_session and self._session_id:
            import threading

            threading.Thread(
                target=self.delete_session, args=(self._session_id,)
            ).start()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session

        Args:
            session_id: Session ID

        Returns:
            bool: True if deletion was successful
        """
        return self.adapter.delete_session(session_id)
