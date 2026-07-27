"""Live APISpace and wttr.in weather adapters."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import requests

from road_distress_agent.error_classifiers import classify_http_error
from road_distress_agent.errors import BoundaryError, ErrorCategory, make_error_info
from road_distress_agent.state import AddressContext, WeatherContext
from road_distress_agent.tools.weather import (
    _dry_window_hours,
    _float_or_none,
    _int_or_none,
    _is_rainy_day,
    _mark_day_suitability,
    _parse_forecast_day,
    _weather_desc,
)


class APISpaceZipLookup:
    def __init__(self, *, token: str | None = None, timeout: int | None = None) -> None:
        self.token = token or os.environ.get("APISPACE_TOKEN")
        self.timeout = timeout or int(os.environ.get("APISPACE_TIMEOUT_SECONDS", "10"))
        self.url = os.environ.get("APISPACE_POSTCODE_URL") or (
            "https://eolink.o.apispace.com/postcode/addr"
        )

    def resolve(self, zip_code: str) -> AddressContext:
        if not self.token:
            raise BoundaryError(_zip_config_missing(zip_code))
        payload = self._post(zip_code)
        items = (payload.get("result") or {}).get("list") or []
        if not items:
            raise BoundaryError(_zip_not_found(zip_code, payload))
        return _address_from_zip_payload(zip_code, payload, items[0])

    def _post(self, zip_code: str) -> dict[str, Any]:
        try:
            response = requests.post(
                self.url,
                data={"postcode": zip_code, "page": 1, "pageSize": 1},
                headers={
                    "X-APISpace-Token": self.token,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            info = classify_http_error(
                exc,
                domain="WEATHER",
                step="ZIP",
                responsibility="邮编解析失败",
                service="APISpace",
                url=self.url,
            )
            raise BoundaryError(info, exc) from exc


class WttrWeatherTool:
    def __init__(self, *, timeout: int | None = None) -> None:
        self.timeout = timeout or int(os.environ.get("WTTR_TIMEOUT_SECONDS", "10"))

    def get_forecast(self, location_query: str) -> WeatherContext:
        url = f"https://wttr.in/{quote(location_query.strip(), safe=',~')}?format=j1"
        payload = self._get(url)
        current = (payload.get("current_condition") or [{}])[0]
        if not payload.get("weather"):
            raise BoundaryError(_forecast_parse_error(location_query, payload))
        days = [
            _mark_day_suitability(_parse_forecast_day(item))
            for item in payload.get("weather", [])[:3]
        ]
        return WeatherContext(
            status="available",
            source="wttr.in",
            location_label=location_query,
            current_temp_c=_float_or_none(current.get("temp_C")),
            current_feels_like_c=_float_or_none(current.get("FeelsLikeC")),
            current_condition=_weather_desc(current),
            humidity=_int_or_none(current.get("humidity")),
            wind_kmph=_float_or_none(current.get("windspeedKmph")),
            precip_mm=_float_or_none(current.get("precipMM")),
            precipitation_expected=any(_is_rainy_day(day) for day in days[:1]),
            dry_window_hours=_dry_window_hours(days),
            forecast_days=days,
            constraints=[reason for day in days[:1] for reason in day.risk_reasons],
            raw_response=payload,
        )

    def _get(self, url: str) -> dict[str, Any]:
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            info = classify_http_error(
                exc,
                domain="WEATHER",
                step="FORECAST",
                responsibility="天气查询失败",
                service="wttr.in",
                url=url,
            )
            raise BoundaryError(info, exc) from exc


def _address_from_zip_payload(
    zip_code: str,
    payload: dict[str, Any],
    item: dict[str, Any],
) -> AddressContext:
    city = item.get("City") or item.get("city")
    state = item.get("Province") or item.get("province") or item.get("State")
    county = item.get("County") or item.get("District") or item.get("county")
    if not city:
        raise BoundaryError(_zip_parse_error(zip_code, payload))
    return AddressContext(
        zip_code=zip_code,
        location_query=city,
        city=city,
        county=county,
        state=state,
        raw_response=payload,
        status="resolved",
    )


def _zip_config_missing(zip_code: str):
    return make_error_info(
        domain="WEATHER",
        step="ZIP",
        category=ErrorCategory.CONFIG_MISSING,
        responsibility="邮编解析未执行",
        reason="未配置 APISPACE_TOKEN",
        hint="在 .env 设置 APISPACE_TOKEN 后重试，或直接提供城市名。",
        raw=f"zip_code={zip_code}",
        retriable=False,
    )


def _zip_not_found(zip_code: str, payload: dict[str, Any]):
    return make_error_info(
        domain="WEATHER",
        step="ZIP",
        category=ErrorCategory.NOT_FOUND,
        responsibility="邮编未解析到城市",
        reason=f"邮编 {zip_code} 无匹配",
        hint="请确认邮编或直接提供城市名。",
        raw=str(payload),
        retriable=False,
    )


def _zip_parse_error(zip_code: str, payload: dict[str, Any]):
    return make_error_info(
        domain="WEATHER",
        step="ZIP",
        category=ErrorCategory.PARSE,
        responsibility="邮编结果无法解析",
        reason=f"APISpace 返回结构缺少 City 字段，zip={zip_code}",
        hint="查看 raw，可能是接口改版或返回结构异常。",
        raw=str(payload),
        retriable=False,
    )


def _forecast_parse_error(location_query: str, payload: dict[str, Any]):
    return make_error_info(
        domain="WEATHER",
        step="FORECAST",
        category=ErrorCategory.PARSE,
        responsibility="天气数据无法解析",
        reason=f'wttr.in 返回空 weather 数组，query="{location_query}"',
        hint="查看 raw，可能是接口改版或城市名不被识别。",
        raw=str(payload),
        retriable=False,
    )
