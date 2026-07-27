"""Vision LLM tool protocol, deterministic mock, and OpenAI-compatible adapter."""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from road_distress_agent.errors import BoundaryError
from road_distress_agent.settings import use_real_vision
from road_distress_agent.vlm_errors import classify_vlm_error, classify_vlm_parse_error


class VisionSceneDescriber(Protocol):
    def describe(
        self, image_path: str, user_text: str | None = None, locale: str = "zh-CN"
    ) -> str: ...


MOCK_SCENES = {
    "crack": (
        "照片显示为沥青路面直线路段，现场没有明显坑槽；裂隙边缘未见明显破碎或"
        "剥落；周围地形平坦开阔，未见明显积水。"
    ),
    "pothole": (
        "照片显示沥青路面坑槽，坑内有积水或潮湿痕迹；坑槽边缘不规则，局部可见"
        "碎石基层；道路处于下坡弯道路段，重载车辆作用风险较高。"
    ),
    "rutting": (
        "照片显示沥青路面存在明显纵向车辙，位置接近红绿灯交叉口停车排队和"
        "制动区域；雨天易形成带状积水，未见单独坑槽形态。"
    ),
}


class MockVisionSceneDescriber:
    def describe(self, image_path: str, user_text: str | None = None, locale: str = "zh-CN") -> str:
        name = Path(image_path).name.lower()
        if locale == "en-US":
            if "pothole" in name or "坑槽" in name:
                return (
                    "The image appears to show an asphalt pavement pothole with wet or "
                    "damp areas inside; the edge is irregular and some granular base "
                    "material may be exposed."
                )
            if "rutting" in name or "rut" in name or "车辙" in name:
                return (
                    "The image appears to show longitudinal rutting on asphalt pavement, "
                    "likely in a braking or queueing area; no separate pothole shape is clear."
                )
            return (
                "The image appears to show an asphalt road segment with visible cracking; "
                "there is no obvious pothole or standing water."
            )
        if "pothole" in name or "坑槽" in name:
            return MOCK_SCENES["pothole"]
        if "rutting" in name or "rut" in name or "车辙" in name:
            return MOCK_SCENES["rutting"]
        return MOCK_SCENES["crack"]


DEFAULT_QWEN_VL_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_VL_MODEL = "qwen3.5-flash"
DEFAULT_MAX_INLINE_IMAGE_BYTES = 7 * 1024 * 1024


@lru_cache(maxsize=1)
def _vlm_prompt_text() -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "vlm_scene_describer.txt"
    return prompt_path.read_text(encoding="utf-8")


def _local_path_from_uri(uri: str) -> Path:
    if uri.startswith("file://"):
        return Path(uri.removeprefix("file://"))
    if uri.startswith("local://"):
        return Path(uri.removeprefix("local://"))
    return Path(uri)


def _image_data_url(image_path: str, *, max_bytes: int) -> str:
    path = _local_path_from_uri(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"Image is {size} bytes, above inline limit {max_bytes}; object storage is not in M6."
        )
    media_type = guess_type(path.name)[0] or "application/octet-stream"
    if not media_type.startswith("image/"):
        raise ValueError(f"Unsupported image media type for VLM: {media_type}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


class OpenAICompatibleVisionSceneDescriber:
    """Call a vision model through an OpenAI-compatible chat completions API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        max_retries: int = 1,
        max_inline_image_bytes: int | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.base_url = base_url or os.environ.get("QWEN_VL_BASE_URL") or DEFAULT_QWEN_VL_BASE_URL
        self.model = model or os.environ.get("QWEN_VL_MODEL") or DEFAULT_QWEN_VL_MODEL
        self.timeout = timeout or int(os.environ.get("QWEN_VL_TIMEOUT_SECONDS", "45"))
        self.max_retries = max_retries
        self.max_inline_image_bytes = max_inline_image_bytes or int(
            os.environ.get("QWEN_VL_MAX_INLINE_IMAGE_BYTES", DEFAULT_MAX_INLINE_IMAGE_BYTES)
        )

    def describe(self, image_path: str, user_text: str | None = None, locale: str = "zh-CN") -> str:
        if not self.api_key:
            exc = RuntimeError("DASHSCOPE_API_KEY is required when ROAD_DISTRESS_USE_REAL_VISION=1")
            raise BoundaryError(classify_vlm_error(exc, step="CONFIG"), exc)

        from langchain_openai import ChatOpenAI

        try:
            image_url = _image_data_url(image_path, max_bytes=self.max_inline_image_bytes)
        except Exception as exc:
            raise BoundaryError(classify_vlm_error(exc, step="IMAGE"), exc) from exc
        user_context = user_text.strip() if user_text else "用户未提供额外文字。"
        llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=0,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        response = self._invoke_llm(llm, image_url, user_context, locale)
        text = _response_text(response.content)
        if not text:
            info = classify_vlm_parse_error(response.content)
            raise BoundaryError(info)
        return text

    def _invoke_llm(self, llm: Any, image_url: str, user_context: str, locale: str):
        messages = [
            SystemMessage(content=_vlm_prompt_text()),
            HumanMessage(content=_vlm_content(image_url, user_context, locale)),
        ]
        try:
            return llm.invoke(messages)
        except Exception as exc:
            info = classify_vlm_error(exc, step="CALL", timeout_seconds=self.timeout)
            raise BoundaryError(info, exc) from exc


def _vlm_content(image_url: str, user_context: str, locale: str) -> list[dict[str, object]]:
    return [
        {
            "type": "text",
            "text": (
                f"target_locale = {locale}\n"
                "Describe visible road distress and site-context facts.\n"
                f"用户文字上下文：{user_context}"
            ),
        },
        {"type": "image_url", "image_url": {"url": image_url}},
    ]


def _response_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {None, "text"}
        ]
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def make_vision_scene_describer() -> VisionSceneDescriber:
    if use_real_vision():
        return OpenAICompatibleVisionSceneDescriber()
    return MockVisionSceneDescriber()
