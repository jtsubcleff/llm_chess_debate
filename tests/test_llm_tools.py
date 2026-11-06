"""Exercise actual SDK serialization and parsing with offline HTTP transports."""

import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from core.chess.tools import CHESS_TOOLS, VALIDATE_MOVE_SEQUENCE
from core.llm_api.anthropic_llm import AnthropicChatModel
from core.llm_api.base_llm import StopReason
from core.llm_api.openai_llm import OpenAIChatModel
from core.llm_api.tool_calls import assistant_message


class NativeToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_tool_round_trip(self):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            tool_call = dict(
                id="c1",
                type="function",
                function=dict(
                    name="validate_move_sequence", arguments='{"moves":["e4"]}'
                ),
            )
            return httpx.Response(
                200,
                headers={
                    "request-id": "req_anthropic_test",
                    "x-request-id": "req_openai_test",
                },
                json=dict(
                    id="test",
                    object="chat.completion",
                    created=0,
                    model="gpt-4",
                    usage=dict(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                    choices=[
                        dict(
                            index=0,
                            finish_reason="tool_calls"
                            if len(requests) == 1
                            else "stop",
                            message=dict(
                                role="assistant", content=None, tool_calls=[tool_call]
                            )
                            if len(requests) == 1
                            else dict(role="assistant", content="e4"),
                        )
                    ],
                ),
            )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            adapter = OpenAIChatModel(0.9, None)
        await adapter.client.close()
        async with AsyncOpenAI(
            api_key="test-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            adapter.client = client
            messages = [dict(role="user", content="Best move?")]
            first = (
                await adapter._make_api_call(
                    messages, "gpt-4", time.time(), tools=CHESS_TOOLS
                )
            )[0]
            self.assertEqual(first.stop_reason, StopReason.TOOL_CALL)
            self.assertEqual((first.input_tokens, first.output_tokens), (20, 10))
            messages.extend(
                [
                    assistant_message(first),
                    dict(role="tool", tool_call_id="c1", content='{"legal":true}'),
                ]
            )
            final = (
                await adapter._make_api_call(
                    messages,
                    "gpt-4",
                    time.time(),
                    tools=CHESS_TOOLS,
                    tool_choice="none",
                )
            )[0]
        self.assertEqual(final.completion, "e4")
        self.assertEqual(first.request_id, "req_openai_test")
        self.assertEqual(final.to_dict()["request_id"], "req_openai_test")
        self.assertEqual(len(requests[0]["tools"]), 1)
        self.assertEqual(requests[1]["messages"][1]["tool_calls"][0]["id"], "c1")
        self.assertEqual(requests[1]["messages"][2]["tool_call_id"], "c1")

    async def test_responses_keeps_reasoning_and_call_id(self):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            output = [
                dict(type="reasoning", id="r1", summary=[]),
                dict(
                    type="function_call",
                    id="fc1",
                    call_id="c1",
                    name="validate_move_sequence",
                    arguments='{"moves":["e4"]}',
                    status="completed",
                ),
            ]
            if len(requests) == 2:
                output = [
                    dict(
                        type="message",
                        id="m1",
                        role="assistant",
                        status="completed",
                        content=[dict(type="output_text", text="e4", annotations=[])],
                    )
                ]
            return httpx.Response(
                200,
                headers={
                    "request-id": "req_anthropic_test",
                    "x-request-id": "req_openai_test",
                },
                json=dict(
                    id="resp1",
                    object="response",
                    created_at=0,
                    model="gpt-6-luna",
                    status="completed",
                    output=output,
                    usage=dict(input_tokens=20, output_tokens=10, total_tokens=30),
                ),
            )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            adapter = OpenAIChatModel(0.9, None)
        await adapter.client.close()
        async with AsyncOpenAI(
            api_key="test-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            adapter.client = client
            messages = [dict(role="user", content="Best move?")]
            first = (
                await adapter._make_api_call(
                    messages,
                    "openai_small",
                    time.time(),
                    tools=CHESS_TOOLS,
                )
            )[0]
            self.assertEqual(first.tool_calls[0]["id"], "c1")
            messages.extend(
                [
                    assistant_message(first),
                    dict(role="tool", tool_call_id="c1", content='{"legal":true}'),
                ]
            )
            final = (
                await adapter._make_api_call(
                    messages,
                    "openai_small",
                    time.time(),
                    tools=CHESS_TOOLS,
                    tool_choice="none",
                )
            )[0]
        self.assertEqual(final.completion, "e4")
        self.assertEqual(first.request_id, "req_openai_test")
        self.assertEqual(final.to_dict()["request_id"], "req_openai_test")
        self.assertEqual(len(requests[0]["tools"]), 1)
        self.assertEqual(requests[0]["tools"][0]["name"], "validate_move_sequence")
        self.assertEqual(requests[1]["input"][1]["type"], "reasoning")
        self.assertEqual(requests[1]["input"][3]["call_id"], "c1")
        self.assertNotIn("response_items", requests[1]["input"][1])

    async def test_anthropic_tool_round_trip_and_usage(self):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            blocks = [
                dict(type="thinking", thinking="", signature="opaque_claude_signature"),
                dict(type="text", text="Try e4"),
                dict(
                    type="tool_use",
                    id="c1",
                    name="validate_move_sequence",
                    input=dict(moves=["e4"]),
                ),
            ]
            if len(requests) == 2:
                blocks = [dict(type="text", text="e4")]
            return httpx.Response(
                200,
                headers={
                    "request-id": "req_anthropic_test",
                    "x-request-id": "req_openai_test",
                },
                json=dict(
                    id="msg1",
                    type="message",
                    role="assistant",
                    model="test-model",
                    content=blocks,
                    stop_reason="tool_use" if len(requests) == 1 else "end_turn",
                    stop_sequence=None,
                    usage=dict(input_tokens=20, output_tokens=10),
                ),
            )

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            adapter = AnthropicChatModel(num_threads=2)
        await adapter.client.close()
        async with AsyncAnthropic(
            api_key="test-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            adapter.client = client
            messages = [
                dict(role="system", content="Solve chess"),
                dict(role="user", content="Best move?"),
            ]
            with patch.object(Path, "write_text") as history:
                first = (
                    await adapter(
                        ["anthropic_small"],
                        messages,
                        False,
                        1,
                        tools=CHESS_TOOLS,
                    )
                )[0]
                messages.extend(
                    [
                        assistant_message(first),
                        dict(role="tool", tool_call_id="c1", content='{"legal":true}'),
                    ]
                )
                final = (
                    await adapter(
                        ["anthropic_small"],
                        messages,
                        False,
                        1,
                        tools=CHESS_TOOLS,
                        tool_choice="none",
                    )
                )[0]
        self.assertEqual(first.stop_reason, StopReason.TOOL_CALL)
        self.assertEqual((first.input_tokens, first.output_tokens), (20, 10))
        self.assertEqual(final.completion, "e4")
        self.assertEqual(first.request_id, "req_anthropic_test")
        self.assertEqual(final.to_dict()["request_id"], "req_anthropic_test")
        self.assertEqual(len(requests[0]["tools"]), 1)
        self.assertEqual(
            requests[0]["tools"][0]["input_schema"],
            VALIDATE_MOVE_SEQUENCE["function"]["parameters"],
        )
        self.assertEqual(requests[1]["messages"][-1]["content"][0]["tool_use_id"], "c1")
        self.assertEqual(requests[1]["tool_choice"], {"type": "none"})
        self.assertEqual(
            requests[1]["messages"][1]["content"][0]["signature"],
            "opaque_claude_signature",
        )
        for saved in history.call_args_list:
            self.assertNotIn("opaque_claude_signature", saved.args[0])


if __name__ == "__main__":
    unittest.main()
