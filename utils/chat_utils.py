import numpy as np
import time
import requests
import json
from arguments import get_args
import base64
import openai
from openai import OpenAI
from io import BytesIO
from pathlib import Path
import threading
import cv2
from dataclasses import asdict, is_dataclass

import utils.visualization as vu
from utils.global_planners.errors import GPTResponseError
import os

def _get_openai_client():
    """Lazy OpenAI client.

    Constructing OpenAI() at module import time crashes with
    'OPENAI_API_KEY' not set, which prevented main.py from running
    even when --nav_mode != 'gpt'. Building the client lazily lets
    every non-gpt path (nearest / co_ut / fill) start without an
    API key.
    """
    global _openai_client
    try:
        _openai_client
    except NameError:
        _openai_client = OpenAI()
    return _openai_client


# Backwards-compat shim: code that does ``client.chat.completions.create``
# transparently triggers the lazy build on first attribute access.
class _LazyClient:
    def __getattr__(self, name):
        return getattr(_get_openai_client(), name)


client = _LazyClient()

gpt_name = [
            'text-davinci-003',
            'gpt-3.5-turbo-0125',
            'gpt-4o',
            'gpt-4o-mini'
        ]           
def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


args = get_args()

def has_display() -> bool:
    """Return True when a display is available for OpenCV GUI windows.

    On Unix-like systems this checks the DISPLAY env var. On Windows
    we assume a display is available.
    """
    return os.name == 'nt' or bool(os.environ.get('DISPLAY'))

def get_all_candidate_maps(target_edge_map, top_view_map, pose):
    # show paths in map
    candidate_map_list = []
    for i in range(int(target_edge_map.max())):
        map_with_frontier = top_view_map.copy()
        path_map = np.zeros(target_edge_map.shape)
        path_map[target_edge_map == i+1] = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        path_map = cv2.dilate((path_map).astype('uint8'), kernel)
        map_with_frontier[path_map == 1] = [255, 0 , 0]
        map_with_frontier = np.flipud(map_with_frontier)
        map_with_pose = vu.write_number(map_with_frontier, pose, i)
        buffered = BytesIO()
        map_with_pose.save(buffered, format="JPEG")
        candidate_map_list.append(buffered)
        
        opencv_image = np.array(map_with_pose)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        if args.visualize and has_display():
            cv2.imshow("candidate_{}".format(i), opencv_image)
            cv2.waitKey(1)
        
    return candidate_map_list

def get_all_candidate_obs_maps(target_edge_map, top_view_map, pose):
    # show paths in map
    candidate_map_list = []
    for i in range(int(target_edge_map.max())):
        map_with_frontier = top_view_map.copy()
        mask = np.any(top_view_map != 0, axis=-1)
        map_with_frontier[mask] = [255, 255, 255]
        path_map = np.zeros(target_edge_map.shape)
        path_map[target_edge_map == i+1] = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        path_map = cv2.dilate((path_map).astype('uint8'), kernel)
        map_with_frontier[path_map == 1] = [255, 0 , 0]
        map_with_frontier = np.flipud(map_with_frontier)
        map_with_pose = vu.write_number(map_with_frontier, pose, i)
        buffered = BytesIO()
        map_with_pose.save(buffered, format="JPEG")
        candidate_map_list.append(buffered)
        
    return candidate_map_list

def get_all_candidate_full_maps(image_id, target_edge_map, top_view_map, pose):
    # show paths in map
    candidate_map_list = []
    for i in range(int(target_edge_map.max())):
        map_with_frontier = top_view_map.copy()
        path_map = np.zeros(target_edge_map.shape)
        path_map[target_edge_map == i+1] = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        path_map = cv2.dilate((path_map).astype('uint8'), kernel)
        map_with_frontier[path_map == 1] = [255, 0 , 0]
        map_with_frontier = np.flipud(map_with_frontier)
        
        frontier_image = image_id[i]
        np_image = np.asarray(frontier_image)
        resized_image1 = cv2.resize(np_image, (480, 480))
        resized_image2 = cv2.resize(np.asarray(map_with_frontier), (480, 480))
        
        combined_image = np.hstack((resized_image2, resized_image1))
        
        map_with_pose = vu.write_number_full(combined_image, pose, i)
        
        if args.visualize and has_display():
            cv2.imshow("map_with_pose_"+str(i), transform_rgb_bgr(np.asarray(map_with_pose)))
            cv2.waitKey(1)
        buffered = BytesIO()
        map_with_pose.save(buffered, format="JPEG")
        candidate_map_list.append(buffered)
        
    return candidate_map_list

def _risk_context_to_json(risk_context):
    """Serialise risk reports without coupling chat code to ``utils.risk``."""

    def _jsonable(value):
        if hasattr(value, "to_dict"):
            return _jsonable(value.to_dict())
        if is_dataclass(value):
            return _jsonable(asdict(value))
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    if isinstance(risk_context, str):
        return risk_context
    return json.dumps(
        _jsonable(risk_context),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )


def message_prepare(
    prompt,
    candidate_map_list,
    navigation_instruct,
    num_agents=None,
):
    """Build the normal, risk-free VLM frontier-assignment request."""
    base64_image_list = []
    for image_candidate in candidate_map_list:
        base64_image_list.append(base64.b64encode(image_candidate.getvalue()).decode("utf-8"))


    message = []
    message.append({"role": "system", "content": prompt})
    image_contents = []
    robot_count = int(args.num_agents if num_agents is None else num_agents)
    if robot_count < 1:
        raise ValueError("num_agents must be at least 1")
    if robot_count == 1:
        task_text = "1 robot needs to find a " + navigation_instruct
    else:
        task_text = f"{robot_count} robots need to find a " + navigation_instruct
    image_contents.append({
        "type": "text",
        "text": task_text,
    })
    for base64_image in base64_image_list:
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    message.append({"role": "user", "content": image_contents})
    
    return message


def risk_message_prepare(
    prompt,
    candidate_map_list,
    navigation_instruct,
    risk_context,
    num_agents=None,
):
    """Build the dedicated risk-aware VLM frontier-assignment request."""
    if risk_context is None:
        raise ValueError("risk_context is required for a risk-aware VLM request")

    base64_image_list = [
        base64.b64encode(image_candidate.getvalue()).decode("utf-8")
        for image_candidate in candidate_map_list
    ]
    robot_count = int(args.num_agents if num_agents is None else num_agents)
    if robot_count < 1:
        raise ValueError("num_agents must be at least 1")

    robot_ids = ", ".join(f"robot_{index}" for index in range(robot_count))
    task_text = (
        "Risk-aware frontier assignment\n"
        f"Robots ({robot_count}): {robot_ids}\n"
        f"Target object: {navigation_instruct}\n"
        f"Candidate frontiers: {len(base64_image_list)} images ordered from "
        "frontier_0 upward\n\n"
        "Hazard report (JSON; match entries to images by frontier_id):\n"
        f"{_risk_context_to_json(risk_context)}"
    )
    image_contents = [{"type": "text", "text": task_text}]
    for base64_image in base64_image_list:
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            },
        })
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": image_contents},
    ]


GPT_MODEL = "gpt-4o"
GPT_MAX_ATTEMPTS = 5
GPT_MAX_COMPLETION_TOKENS = 300
_GPT_RESPONSE_STATUS_LOCK = threading.Lock()


def _frontier_assignment_response_format(num_agents, num_frontiers):
    """Build the strict schema for this exact robot/frontier request."""
    if int(num_agents) < 1:
        raise ValueError("num_agents must be at least 1")
    if int(num_frontiers) < 1:
        raise ValueError("num_frontiers must be at least 1")

    frontier_values = [
        f"frontier_{frontier_id}"
        for frontier_id in range(int(num_frontiers))
    ]
    properties = {
        f"robot_{robot_id}": {
            "type": "string",
            "enum": frontier_values,
        }
        for robot_id in range(int(num_agents))
    }
    properties["reason"] = {"type": "string"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "frontier_assignment",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def _jsonable_response_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(key): _jsonable_response_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_response_value(item) for item in value]
    return str(value)


def _emit_gpt_response_status(payload):
    """Append one credential-free JSONL record without polluting stdout."""
    rank = int(getattr(args, "rank", 0))
    rank_suffix = "" if rank == 0 else f"_rank{rank}"
    status_path = (
        Path(args.dump_location)
        / "logs"
        / "gpt"
        / f"gpt_response_status{rank_suffix}.jsonl"
    )
    record = dict(_jsonable_response_value(payload))
    record.setdefault("logged_at_unix_s", time.time())
    record.setdefault("process_id", os.getpid())
    line = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    with _GPT_RESPONSE_STATUS_LOCK:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _validate_frontier_assignment(
    payload,
    *,
    num_agents,
    num_frontiers,
):
    if not isinstance(payload, dict):
        return "response is not a JSON object"
    required = {
        *(f"robot_{robot_id}" for robot_id in range(int(num_agents))),
        "reason",
    }
    if set(payload) != required:
        return (
            "response keys do not match schema: expected "
            + ", ".join(sorted(required))
        )
    if not isinstance(payload["reason"], str):
        return "reason must be a string"
    for robot_id in range(int(num_agents)):
        key = f"robot_{robot_id}"
        expected = {
            f"frontier_{frontier_id}"
            for frontier_id in range(int(num_frontiers))
        }
        if payload[key] not in expected:
            return f"{key} is outside the current frontier set"
    return None


def _api_error_status(error, *, attempt, num_agents, num_frontiers):
    return {
        "event": "gpt_response",
        "attempt": int(attempt),
        "max_attempts": GPT_MAX_ATTEMPTS,
        "model_requested": GPT_MODEL,
        "num_agents": int(num_agents),
        "num_frontiers": int(num_frontiers),
        "outcome": "api_error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "status_code": getattr(error, "status_code", None),
        "request_id": getattr(error, "request_id", None),
        "error_code": getattr(error, "code", None),
    }


def chat_with_gpt4v(
    chat_history,
    gpt_type=args.gpt_type,
    *,
    num_agents=None,
    num_frontiers=None,
):
    """Return a strictly validated assignment or raise ``GPTResponseError``."""
    del gpt_type  # Keep the historical call signature; this baseline is GPT-4o.
    agent_count = int(args.num_agents if num_agents is None else num_agents)
    frontier_count = int(
        len(chat_history[1]["content"]) - 1
        if num_frontiers is None
        else num_frontiers
    )
    response_format = _frontier_assignment_response_format(
        agent_count,
        frontier_count,
    )
    last_outcome = "no_response"

    for attempt in range(1, GPT_MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                response_format=response_format,
                messages=chat_history,
                temperature=0.1,
                max_completion_tokens=GPT_MAX_COMPLETION_TOKENS,
            )
        except (openai.APIConnectionError, openai.RateLimitError) as error:
            status = _api_error_status(
                error,
                attempt=attempt,
                num_agents=agent_count,
                num_frontiers=frontier_count,
            )
            status["outcome"] = (
                "connection_error"
                if isinstance(error, openai.APIConnectionError)
                else "rate_limit"
            )
            _emit_gpt_response_status(status)
            last_outcome = status["outcome"]
            if attempt < GPT_MAX_ATTEMPTS:
                time.sleep(min(2 ** (attempt - 1), 8))
            continue
        except openai.APIError as error:
            status = _api_error_status(
                error,
                attempt=attempt,
                num_agents=agent_count,
                num_frontiers=frontier_count,
            )
            _emit_gpt_response_status(status)
            last_outcome = status["outcome"]
            if attempt < GPT_MAX_ATTEMPTS:
                time.sleep(min(2 ** (attempt - 1), 8))
            continue

        choice = response.choices[0]
        message = choice.message
        content = message.content
        refusal = getattr(message, "refusal", None)
        finish_reason = choice.finish_reason
        if content is not None:
            print(f"{GPT_MODEL} response:", flush=True)
            print(content, flush=True)
        status = {
            "event": "gpt_response",
            "attempt": attempt,
            "max_attempts": GPT_MAX_ATTEMPTS,
            "response_id": getattr(response, "id", None),
            "request_id": getattr(response, "_request_id", None),
            "model_requested": GPT_MODEL,
            "model_returned": getattr(response, "model", None),
            "created": getattr(response, "created", None),
            "system_fingerprint": getattr(
                response, "system_fingerprint", None
            ),
            "service_tier": getattr(response, "service_tier", None),
            "finish_reason": finish_reason,
            "refusal": refusal,
            "content": content,
            "content_present": content is not None,
            "content_chars": 0 if content is None else len(content),
            "tool_call_count": len(getattr(message, "tool_calls", None) or []),
            "usage": getattr(response, "usage", None),
            "num_agents": agent_count,
            "num_frontiers": frontier_count,
        }

        if refusal:
            status["outcome"] = "refusal"
            _emit_gpt_response_status(status)
            raise GPTResponseError(
                "GPT refused the frontier-assignment request",
                reason="refusal",
                attempts=attempt,
            )
        if finish_reason == "content_filter":
            status["outcome"] = "content_filter"
            _emit_gpt_response_status(status)
            raise GPTResponseError(
                "GPT frontier assignment was blocked by the content filter",
                reason="content_filter",
                attempts=attempt,
            )
        if content is None:
            status["outcome"] = "empty_content"
            _emit_gpt_response_status(status)
            last_outcome = status["outcome"]
            continue
        if finish_reason == "length":
            status["outcome"] = "truncated"
            _emit_gpt_response_status(status)
            last_outcome = status["outcome"]
            continue

        try:
            assignment = json.loads(content)
        except json.JSONDecodeError as error:
            status["outcome"] = "invalid_json"
            status["validation_error"] = str(error)
            _emit_gpt_response_status(status)
            last_outcome = status["outcome"]
            continue

        validation_error = _validate_frontier_assignment(
            assignment,
            num_agents=agent_count,
            num_frontiers=frontier_count,
        )
        if validation_error is not None:
            status["outcome"] = "invalid_schema"
            status["validation_error"] = validation_error
            _emit_gpt_response_status(status)
            last_outcome = status["outcome"]
            continue

        status["outcome"] = "valid"
        _emit_gpt_response_status(status)
        return assignment

    raise GPTResponseError(
        (
            "GPT did not produce a valid frontier assignment after "
            f"{GPT_MAX_ATTEMPTS} attempts"
        ),
        reason=last_outcome,
        attempts=GPT_MAX_ATTEMPTS,
    )
