"""Regressions for auditable GPT frontier-assignment responses."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


with mock.patch.object(sys, "argv", ["test_gpt_response_handling"]):
    from utils import chat_utils

from utils.global_planners.errors import GPTResponseError


class _Usage:
    def model_dump(self, mode=None):
        del mode
        return {
            "prompt_tokens": 100,
            "completion_tokens": 25,
            "total_tokens": 125,
        }


def _response(
    content,
    *,
    finish_reason="stop",
    refusal=None,
    response_id="chatcmpl-test",
):
    return SimpleNamespace(
        id=response_id,
        _request_id="req-test",
        model="gpt-4o-2024-08-06",
        created=1785952800,
        system_fingerprint="fp_test",
        service_tier="default",
        usage=_Usage(),
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    refusal=refusal,
                    tool_calls=None,
                ),
            )
        ],
    )


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class GPTResponseHandlingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self._dump_patch = mock.patch.object(
            chat_utils.args,
            "dump_location",
            self._temporary_directory.name,
        )
        self._dump_patch.start()
        self.addCleanup(self._dump_patch.stop)
        self.status_path = (
            Path(self._temporary_directory.name)
            / "logs"
            / "gpt"
            / "gpt_response_status.jsonl"
        )

    def _read_statuses(self):
        if not self.status_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.status_path.read_text(encoding="utf-8").splitlines()
        ]

    def _call(self, responses, *, num_agents=2, num_frontiers=3):
        completions = _FakeCompletions(responses)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        stream = io.StringIO()
        with mock.patch.object(chat_utils, "client", fake_client):
            with redirect_stdout(stream):
                result = chat_utils.chat_with_gpt4v(
                    [{"role": "system", "content": "prompt"}],
                    num_agents=num_agents,
                    num_frontiers=num_frontiers,
                )
        return result, completions.calls, self._read_statuses(), stream.getvalue()

    def test_valid_response_uses_strict_dynamic_schema_and_larger_budget(
        self,
    ) -> None:
        payload = {
            "robot_0": "frontier_2",
            "robot_1": "frontier_0",
            "reason": "balanced coverage",
        }
        result, calls, statuses, stdout = self._call(
            [_response(json.dumps(payload))]
        )

        self.assertEqual(result, payload)
        self.assertEqual(
            stdout,
            f"gpt-4o response:\n{json.dumps(payload)}\n",
        )
        self.assertEqual(len(calls), 1)
        request = calls[0]
        self.assertEqual(request["model"], "gpt-4o")
        self.assertEqual(request["max_completion_tokens"], 300)
        self.assertNotIn("max_tokens", request)
        response_format = request["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        schema = response_format["json_schema"]["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["required"],
            ["robot_0", "robot_1", "reason"],
        )
        self.assertEqual(
            schema["properties"]["robot_0"]["enum"],
            ["frontier_0", "frontier_1", "frontier_2"],
        )

        self.assertEqual(len(statuses), 1)
        status = statuses[0]
        self.assertEqual(status["outcome"], "valid")
        self.assertEqual(status["response_id"], "chatcmpl-test")
        self.assertEqual(status["request_id"], "req-test")
        self.assertEqual(status["finish_reason"], "stop")
        self.assertEqual(status["usage"]["total_tokens"], 125)
        self.assertEqual(status["content"], json.dumps(payload))
        self.assertIsInstance(status["logged_at_unix_s"], float)
        self.assertIsInstance(status["process_id"], int)

    def test_empty_content_is_logged_and_retried(self) -> None:
        payload = {
            "robot_0": "frontier_0",
            "robot_1": "frontier_1",
            "reason": "retry succeeded",
        }
        result, calls, statuses, stdout = self._call(
            [
                _response(None, response_id="chatcmpl-empty"),
                _response(json.dumps(payload)),
            ]
        )

        self.assertEqual(result, payload)
        self.assertEqual(
            stdout,
            f"gpt-4o response:\n{json.dumps(payload)}\n",
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [status["outcome"] for status in statuses],
            ["empty_content", "valid"],
        )
        self.assertFalse(statuses[0]["content_present"])
        self.assertEqual(statuses[0]["content_chars"], 0)

    def test_repeated_empty_content_raises_typed_terminal_failure(
        self,
    ) -> None:
        completions = _FakeCompletions(
            [
                _response(None, response_id=f"chatcmpl-empty-{attempt}")
                for attempt in range(5)
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        stream = io.StringIO()

        with mock.patch.object(chat_utils, "client", fake_client):
            with redirect_stdout(stream):
                with self.assertRaises(GPTResponseError) as caught:
                    chat_utils.chat_with_gpt4v(
                        [],
                        num_agents=2,
                        num_frontiers=2,
                    )

        self.assertEqual(caught.exception.reason, "empty_content")
        self.assertEqual(caught.exception.attempts, 5)
        self.assertEqual(len(completions.calls), 5)
        self.assertEqual(stream.getvalue(), "")
        self.assertEqual(len(self._read_statuses()), 5)

    def test_refusal_is_logged_and_raises_without_retry(self) -> None:
        completions = _FakeCompletions(
            [_response(None, refusal="cannot comply")]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        stream = io.StringIO()

        with mock.patch.object(chat_utils, "client", fake_client):
            with redirect_stdout(stream):
                with self.assertRaises(GPTResponseError) as caught:
                    chat_utils.chat_with_gpt4v(
                        [],
                        num_agents=2,
                        num_frontiers=2,
                    )

        self.assertEqual(caught.exception.reason, "refusal")
        self.assertEqual(caught.exception.attempts, 1)
        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(stream.getvalue(), "")
        status = self._read_statuses()[0]
        self.assertEqual(status["outcome"], "refusal")
        self.assertEqual(status["refusal"], "cannot comply")

    def test_content_filter_is_logged_and_raises_without_retry(self) -> None:
        completions = _FakeCompletions(
            [_response(None, finish_reason="content_filter")]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        stream = io.StringIO()

        with mock.patch.object(chat_utils, "client", fake_client):
            with redirect_stdout(stream):
                with self.assertRaises(GPTResponseError) as caught:
                    chat_utils.chat_with_gpt4v(
                        [],
                        num_agents=2,
                        num_frontiers=2,
                    )

        self.assertEqual(caught.exception.reason, "content_filter")
        self.assertEqual(caught.exception.attempts, 1)
        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(stream.getvalue(), "")
        status = self._read_statuses()[0]
        self.assertEqual(status["outcome"], "content_filter")
        self.assertIsNone(status["refusal"])


if __name__ == "__main__":
    unittest.main()
