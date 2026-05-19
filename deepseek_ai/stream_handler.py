"""DeepSeek Stream Response Handler
Converts DeepSeek SSE stream to OpenAI compatible format
"""

import json
import re
from typing import Dict, Any, Optional, Callable, Generator
from .tool_parser import ToolParser


class DeepSeekStreamHandler:
    """Handle DeepSeek stream responses and convert to OpenAI format"""
    
    def __init__(
        self,
        model: str,
        session_id: str,
        on_end: Optional[Callable] = None,
        web_search_enabled: bool = False,
        reasoning_effort: Optional[str] = None
    ):
        self.model = model
        self.session_id = session_id
        self.on_end = on_end
        self.web_search_enabled = web_search_enabled
        self.reasoning_effort = reasoning_effort
        self.is_first_chunk = True
        self.message_id = ''
        self.current_path = ''
        self.search_results = []
        self.thinking_started = False
        self.accumulated_token_usage = 2
        self.created = int(__import__('time').time())
        self.tool_call_buffer = ''
        self.has_tool_call = False
    
    def _create_chunk(self, delta: Dict, finish_reason: Optional[str] = None) -> str:
        """Create an OpenAI format chunk"""
        return json.dumps({
            'id': f'{self.session_id}@{self.message_id}',
            'model': self.model,
            'object': 'chat.completion.chunk',
            'choices': [{
                'index': 0,
                'delta': delta,
                'finish_reason': finish_reason or None,
            }],
            'created': self.created,
        })
    
    def handle_stream(self, response) -> Generator[str, None, None]:
        """Handle streaming response"""
        is_thinking_model = (
            'think' in self.model.lower() or 
            'r1' in self.model.lower() or 
            bool(self.reasoning_effort)
        )
        is_silent_model = 'silent' in self.model.lower()
        is_fold_model = (
            'fold' in self.model.lower() or 
            'search' in self.model.lower() or 
            self.web_search_enabled
        ) and not is_thinking_model
        is_search_silent_model = 'search-silent' in self.model.lower()
        
        buffer = ''
        
        for line in response.iter_lines():
            if not line:
                continue
            
            line_str = line if isinstance(line, str) else line.decode('utf-8')
            if not line_str.startswith('data:'):
                continue
            
            data = line_str[5:].strip()
            if data == '[DONE]':
                break
            
            try:
                parsed = json.loads(data)
                chunk = self._process_chunk(
                    parsed, 
                    is_thinking_model, 
                    is_silent_model, 
                    is_fold_model, 
                    is_search_silent_model
                )
                if chunk:
                    yield f'data: {chunk}\n\n'
            except json.JSONDecodeError:
                continue
        
        # Send final chunk
        final_chunk = self._handle_done(is_fold_model, is_search_silent_model)
        if final_chunk:
            yield f'data: {final_chunk}\n\n'
        yield 'data: [DONE]\n\n'
        
        if self.on_end:
            self.on_end()
    
    def _process_chunk(
        self, 
        chunk: Dict, 
        is_thinking_model: bool,
        is_silent_model: bool,
        is_fold_model: bool,
        is_search_silent_model: bool
    ) -> Optional[str]:
        """Process a single chunk from DeepSeek"""
        # Get message ID
        if chunk.get('response_message_id') and not self.message_id:
            self.message_id = chunk['response_message_id']
            return None
        
        previous_path = self.current_path
        
        # Handle response with fragments
        if chunk.get('v') and isinstance(chunk['v'], dict) and chunk['v'].get('response'):
            response_data = chunk['v']['response']
            is_thinking_now = response_data.get('thinking_enabled')
            self.current_path = 'thinking' if is_thinking_now else 'content'
            
            fragments = response_data.get('fragments', [])
            if fragments:
                for fragment in fragments:
                    content = fragment.get('content', '')
                    fragment_type = fragment.get('type', '')
                    
                    if fragment_type == 'THINK':
                        return self._send_content(
                            content, 'thinking', 
                            is_silent_model, is_fold_model, is_search_silent_model
                        )
                    elif fragment_type in ['ANSWER', 'RESPONSE']:
                        return self._send_content(
                            content, 'content',
                            is_silent_model, is_fold_model, is_search_silent_model
                        )
        
        # Handle response/fragments path
        elif chunk.get('p') == 'response/fragments':
            fragments = chunk.get('v', [])
            if isinstance(fragments, list):
                for fragment in fragments:
                    content = fragment.get('content', '')
                    fragment_type = fragment.get('type', '')
                    
                    if fragment_type == 'THINK':
                        self.current_path = 'thinking'
                        return self._send_content(
                            content, 'thinking',
                            is_silent_model, is_fold_model, is_search_silent_model
                        )
                    elif fragment_type in ['ANSWER', 'RESPONSE']:
                        self.current_path = 'content'
                        return self._send_content(
                            content, 'content',
                            is_silent_model, is_fold_model, is_search_silent_model
                        )
        
        # Handle search results
        elif chunk.get('p') == 'response/search_results':
            results = chunk.get('v', [])
            if isinstance(results, list) and chunk.get('o') != 'BATCH':
                self.search_results = results
            return None
        
        # Handle accumulated token usage
        elif chunk.get('p') == 'response' and isinstance(chunk.get('v'), list):
            for item in chunk['v']:
                if item.get('p') == 'accumulated_token_usage' and isinstance(item.get('v'), (int, float)):
                    self.accumulated_token_usage = int(item['v'])
            return None
        
        # Handle simple string content
        content = ''
        if isinstance(chunk.get('v'), str):
            content = chunk['v']
        elif isinstance(chunk.get('v'), list):
            content = ''.join(
                item.get('content', '') 
                for item in chunk['v'] 
                if isinstance(item, dict)
            )
        
        if not content:
            return None
        
        # Determine effective path
        effective_path = self.current_path
        if not effective_path and is_thinking_model:
            effective_path = 'thinking'
        
        return self._send_content(
            content, effective_path,
            is_silent_model, is_fold_model, is_search_silent_model
        )
    
    def _send_content(
        self,
        content: str,
        path: str,
        is_silent_model: bool,
        is_fold_model: bool,
        is_search_silent_model: bool
    ) -> Optional[str]:
        """Send content in OpenAI format"""
        # Clean content
        cleaned = content.replace('FINISHED', '')
        cleaned = re.sub(r'^(SEARCH|WEB_SEARCH|SEARCHING)\s*', '', cleaned, flags=re.IGNORECASE)
        
        # Handle citations
        if is_search_silent_model:
            cleaned = re.sub(r'\[citation:(\d+)\]', '', cleaned)
        else:
            cleaned = re.sub(r'\[citation:(\d+)\]', r'[\1]', cleaned)
        
        if not cleaned:
            return None
        
        # Check for tool calls
        if path in ['content', '']:
            self.tool_call_buffer += cleaned
            tool_calls = ToolParser.parse_tool_calls_from_text(self.tool_call_buffer)
            if tool_calls:
                self.has_tool_call = True
                delta = {'tool_calls': tool_calls}
                if self.is_first_chunk:
                    delta['role'] = 'assistant'
                    self.is_first_chunk = False
                return self._create_chunk(delta)
        
        # Build delta
        delta = {}
        
        if self.is_first_chunk:
            delta['role'] = 'assistant'
            self.is_first_chunk = False
        
        if path == 'thinking':
            if is_silent_model:
                return None
            
            if is_fold_model:
                if not self.thinking_started:
                    self.thinking_started = True
                    delta['content'] = f'<details><summary>Thinking Process</summary><pre>{cleaned}'
                else:
                    delta['content'] = cleaned
            else:
                if cleaned:
                    delta['reasoning_content'] = cleaned
                else:
                    return None
        elif path == 'content':
            if is_fold_model and self.thinking_started:
                delta['content'] = f'</pre></details>{cleaned}'
                self.thinking_started = False
            else:
                delta['content'] = cleaned
        else:
            delta['content'] = cleaned
        
        if delta:
            return self._create_chunk(delta)
        return None
    
    def _handle_done(self, is_fold_model: bool, is_search_silent_model: bool) -> Optional[str]:
        """Handle stream end"""
        delta = {}
        
        if is_fold_model and self.thinking_started:
            delta['content'] = '</pre></details>'
        
        # Add citations if available
        if self.search_results and not is_search_silent_model:
            citations = []
            for result in sorted(self.search_results, key=lambda x: x.get('cite_index', 0)):
                cite_index = result.get('cite_index')
                title = result.get('title', '')
                url = result.get('url', '')
                if cite_index:
                    citations.append(f'[{cite_index}]: [{title}]({url})')
            
            if citations:
                citation_text = '\n\n' + '\n'.join(citations)
                if 'content' in delta:
                    delta['content'] += citation_text
                else:
                    delta['content'] = citation_text
        
        finish_reason = 'tool_calls' if self.has_tool_call else 'stop'
        
        if delta:
            return self._create_chunk(delta, finish_reason)
        else:
            return self._create_chunk({}, finish_reason)
    
    def handle_non_stream(self, response) -> Dict:
        """Handle non-streaming response"""
        accumulated_content = ''
        accumulated_thinking = ''
        message_id = ''
        current_path = ''
        
        is_thinking_model = (
            'think' in self.model.lower() or 
            'r1' in self.model.lower() or 
            bool(self.reasoning_effort)
        )
        is_fold_model = (
            'fold' in self.model.lower() or 
            'search' in self.model.lower() or 
            self.web_search_enabled
        ) and not is_thinking_model
        is_search_silent_model = 'search-silent' in self.model.lower()
        
        buffer = ''
        
        for line in response.iter_lines():
            if not line:
                continue
            
            line_str = line if isinstance(line, str) else line.decode('utf-8')
            if not line_str.startswith('data:'):
                continue
            
            data = line_str[5:].strip()
            if data == '[DONE]':
                break
            
            try:
                parsed = json.loads(data)
                
                if parsed.get('response_message_id') and not message_id:
                    message_id = parsed['response_message_id']
                
                # Handle response with fragments
                if parsed.get('v') and isinstance(parsed['v'], dict) and parsed['v'].get('response'):
                    response_data = parsed['v']['response']
                    is_thinking_now = response_data.get('thinking_enabled')
                    if is_thinking_now is not None:
                        current_path = 'thinking' if is_thinking_now else 'content'
                    
                    fragments = response_data.get('fragments', [])
                    for fragment in fragments:
                        content = fragment.get('content', '')
                        content = content.replace('FINISHED', '')
                        content = re.sub(r'^(SEARCH|WEB_SEARCH|SEARCHING)\s*', '', content, flags=re.IGNORECASE)
                        
                        if fragment.get('type') == 'THINK':
                            accumulated_thinking += content
                        elif fragment.get('type') in ['ANSWER', 'RESPONSE']:
                            accumulated_content += content
                
                # Handle response/fragments path
                elif parsed.get('p') == 'response/fragments':
                    fragments = parsed.get('v', [])
                    if isinstance(fragments, list):
                        for fragment in fragments:
                            content = fragment.get('content', '')
                            content = content.replace('FINISHED', '')
                            content = re.sub(r'^(SEARCH|WEB_SEARCH|SEARCHING)\s*', '', content, flags=re.IGNORECASE)
                            
                            if fragment.get('type') == 'THINK':
                                current_path = 'thinking'
                                accumulated_thinking += content
                            elif fragment.get('type') in ['ANSWER', 'RESPONSE']:
                                current_path = 'content'
                                accumulated_content += content
                
                # Handle simple content
                elif isinstance(parsed.get('v'), str):
                    content = parsed['v'].replace('FINISHED', '')
                    content = re.sub(r'^(SEARCH|WEB_SEARCH|SEARCHING)\s*', '', content, flags=re.IGNORECASE)
                    
                    if not current_path and is_thinking_model:
                        current_path = 'thinking'
                    if not current_path and is_fold_model:
                        current_path = 'content'
                    
                    if current_path == 'thinking':
                        accumulated_thinking += content
                    else:
                        accumulated_content += content
                
                # Handle accumulated token usage
                elif parsed.get('p') == 'response' and isinstance(parsed.get('v'), list):
                    for item in parsed['v']:
                        if item.get('accumulated_token_usage') and isinstance(item.get('v'), (int, float)):
                            self.accumulated_token_usage = int(item['v'])
            
            except json.JSONDecodeError:
                continue
        
        # Parse tool calls from content
        clean_content, tool_calls = ToolParser.parse_tool_calls_from_text_with_content(accumulated_content)
        
        message = {
            'role': 'assistant',
            'content': clean_content.strip() if not tool_calls else None,
        }
        
        if accumulated_thinking.strip():
            message['reasoning_content'] = accumulated_thinking.strip()
        
        if tool_calls:
            message['tool_calls'] = tool_calls
        
        return {
            'id': f'{self.session_id}@{message_id}',
            'model': self.model,
            'object': 'chat.completion',
            'choices': [{
                'index': 0,
                'message': message,
                'finish_reason': 'tool_calls' if tool_calls else 'stop',
            }],
            'usage': {
                'prompt_tokens': 1,
                'completion_tokens': 1,
                'total_tokens': self.accumulated_token_usage
            },
            'created': self.created,
        }
