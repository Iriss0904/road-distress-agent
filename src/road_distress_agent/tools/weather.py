"""Zip and weather tool protocols, deterministic mocks, and HTTP adapters."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from road_distress_agent.settings import use_real_weather
from road_distress_agent.state import (
    AddressContext,
    WeatherContext,
    WeatherForecastDay,
    WeatherForecastHour,
)


class ZipLookupTool(Protocol):
    def resolve(self, zip_code: str) -> AddressContext: ...


class WeatherTool(Protocol):
    def get_forecast(self, location_query: str) -> WeatherContext: ...


def use_real_weather_tools() -> bool:
    return use_real_weather()


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    numeric = _float_or_none(value)
    if numeric is None:
        return None
    return int(numeric)


def _weather_desc(item: dict[str, Any]) -> str | None:
    descriptions = item.get("weatherDesc") or []
    if descriptions and isinstance(descriptions[0], dict):
        value = descriptions[0].get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _is_rainy_day(day: WeatherForecastDay) -> bool:
    if day.total_precip_mm is not None and day.total_precip_mm >= 1.0:
        return True
    for hour in day.hourly:
        if hour.chance_of_rain is not None and hour.chance_of_rain >= 50:
            return True
        if hour.precip_mm is not None and hour.precip_mm >= 0.5:
            return True
    return False


def _is_extreme_day(day: WeatherForecastDay) -> bool:
    if day.max_temp_c is not None and day.max_temp_c >= 35:
        return True
    if day.min_temp_c is not None and day.min_temp_c <= 0:
        return True
    if day.max_wind_kmph is not None and day.max_wind_kmph >= 38:
        return True
    return False


def _mark_day_suitability(day: WeatherForecastDay) -> WeatherForecastDay:
    reasons: list[str] = []
    if _is_rainy_day(day):
        reasons.append("预报有降水，影响基层干燥、层间黏结和压实质量")
    if day.max_temp_c is not None and day.max_temp_c >= 35:
        reasons.append("最高气温较高，应避开正午高温时段")
    if day.min_temp_c is not None and day.min_temp_c <= 0:
        reasons.append("最低气温接近或低于 0°C，需注意结冰、防滑和材料施工温度")
    if day.max_wind_kmph is not None and day.max_wind_kmph >= 38:
        reasons.append("风力较大，需加强围挡、扬尘控制和交通组织")
    return day.model_copy(update={"suitable_for_formal_work": not reasons, "risk_reasons": reasons})


def _dry_window_hours(days: list[WeatherForecastDay]) -> int:
    hours = 0
    for day in days:
        for hour in day.hourly:
            rainy = (hour.chance_of_rain is not None and hour.chance_of_rain >= 50) or (
                hour.precip_mm is not None and hour.precip_mm >= 0.5
            )
            if rainy:
                return hours
            hours += 3
    return hours


class MockZipLookup:
    def resolve(self, zip_code: str) -> AddressContext:
        if zip_code == "130000":
            return AddressContext(
                zip_code="130000",
                location_query="Demo City",
                city="Demo City",
                state="Demo Province",
                status="resolved",
            )
        if zip_code == "511400":
            return AddressContext(
                zip_code="511400",
                location_query="广州",
                city="广州",
                state="广东省",
                status="resolved",
            )
        return AddressContext(zip_code=zip_code, location_query=zip_code, status="unknown")


class MockWeatherTool:
    def get_forecast(self, location_query: str) -> WeatherContext:
        normalized = location_query.strip().lower()
        if normalized in {"130000", "demo city", "demo province demo city"}:
            return _mock_rainy_weather(location_query)
        return _mock_dry_weather(location_query)


def _mock_rainy_weather(location_query: str) -> WeatherContext:
    rainy_today, better_day = _mock_rainy_days()
    return WeatherContext(
        status="available",
        source="mock",
        location_label=location_query,
        current_temp_c=20,
        current_feels_like_c=20,
        current_condition="暴雨",
        humidity=96,
        wind_kmph=12,
        precip_mm=8,
        precipitation_expected=True,
        dry_window_hours=0,
        forecast_days=[rainy_today, better_day],
        constraints=[
            "未来 24 小时有明显降雨，不满足正式修补对干燥基层和层间黏结的施工条件",
            "建议先排水、围挡和临时保通，等待天气窗口改善后再正式施工",
        ],
    )


def _mock_dry_weather(location_query: str) -> WeatherContext:
    dry_day = _mark_day_suitability(
        WeatherForecastDay(
            date=datetime.now(timezone.utc).date().isoformat(),
            min_temp_c=16,
            max_temp_c=26,
            total_precip_mm=0,
            max_wind_kmph=12,
            hourly=[
                WeatherForecastHour(
                    time="1200",
                    temp_c=24,
                    chance_of_rain=5,
                    precip_mm=0,
                    wind_kmph=8,
                    humidity=50,
                    description="晴",
                )
            ],
        )
    )
    return WeatherContext(
        status="available",
        source="mock",
        location_label=location_query,
        current_temp_c=24,
        current_feels_like_c=24,
        current_condition="晴",
        humidity=50,
        wind_kmph=8,
        precip_mm=0,
        precipitation_expected=False,
        dry_window_hours=24,
        forecast_days=[dry_day],
        constraints=[],
    )


def _mock_rainy_days() -> tuple[WeatherForecastDay, WeatherForecastDay]:
    rainy_today = WeatherForecastDay(
        date="2026-05-13",
        min_temp_c=18,
        max_temp_c=23,
        total_precip_mm=22,
        max_wind_kmph=18,
        hourly=[
            WeatherForecastHour(
                time="0900",
                temp_c=20,
                chance_of_rain=95,
                precip_mm=8,
                wind_kmph=12,
                humidity=96,
                description="暴雨",
            )
        ],
    )
    better_day = WeatherForecastDay(
        date="2026-05-15",
        min_temp_c=19,
        max_temp_c=27,
        total_precip_mm=0,
        max_wind_kmph=12,
        hourly=[
            WeatherForecastHour(
                time="1200",
                temp_c=25,
                chance_of_rain=5,
                precip_mm=0,
                wind_kmph=8,
                humidity=55,
                description="晴",
            )
        ],
    )
    return _mark_day_suitability(rainy_today), _mark_day_suitability(better_day)


def _parse_forecast_day(item: dict[str, Any]) -> WeatherForecastDay:
    hourly = [
        WeatherForecastHour(
            time=str(hour.get("time")) if hour.get("time") is not None else None,
            temp_c=_float_or_none(hour.get("tempC")),
            feels_like_c=_float_or_none(hour.get("FeelsLikeC")),
            chance_of_rain=_int_or_none(hour.get("chanceofrain")),
            chance_of_snow=_int_or_none(hour.get("chanceofsnow")),
            precip_mm=_float_or_none(hour.get("precipMM")),
            wind_kmph=_float_or_none(hour.get("windspeedKmph")),
            humidity=_int_or_none(hour.get("humidity")),
            description=_weather_desc(hour),
        )
        for hour in item.get("hourly", [])
    ]
    return WeatherForecastDay(
        date=str(item.get("date")),
        min_temp_c=_float_or_none(item.get("mintempC")),
        max_temp_c=_float_or_none(item.get("maxtempC")),
        total_precip_mm=(
            _float_or_none(item.get("totalSnow_cm"))
            or _float_or_none(item.get("precipMM"))
            or sum(hour.precip_mm or 0 for hour in hourly)
        ),
        max_wind_kmph=_float_or_none(item.get("maxwindKmph")),
        hourly=hourly,
    )


def make_zip_lookup_tool() -> ZipLookupTool:
    if use_real_weather_tools():
        from road_distress_agent.tools.weather_live import APISpaceZipLookup

        return APISpaceZipLookup()
    return MockZipLookup()


def make_weather_tool() -> WeatherTool:
    if use_real_weather_tools():
        from road_distress_agent.tools.weather_live import WttrWeatherTool

        return WttrWeatherTool()
    return MockWeatherTool()
