"""Load address and weather context from zipcode or user-provided city."""

from __future__ import annotations

from road_distress_agent.enums import WorkflowPhase
from road_distress_agent.errors import BoundaryError, ErrorCategory, make_error_info
from road_distress_agent.state import (
    AddressContext,
    AgentState,
    AuditEvent,
    ErrorEvent,
    WeatherContext,
)
from road_distress_agent.tools.weather import make_weather_tool, make_zip_lookup_tool


def _location_query(address: AddressContext | None) -> str | None:
    if address is None:
        return None
    return address.location_query or address.city or address.zip_code


def address_weather_loader(state: AgentState) -> AgentState:
    address = state.get("address_context")
    english = state.get("locale") == "en-US"
    if address is None:
        return _no_address_delta(english)

    resolved_address, errors = _resolve_address(address)
    query = _location_query(resolved_address)
    if not query or resolved_address.status in {"failed", "unknown"}:
        return _unresolved_address_delta(resolved_address, query, errors, english)

    weather, weather_errors = _load_weather(query, english)
    return _weather_delta(resolved_address, weather, query, [*errors, *weather_errors])


def _no_address_delta(english: bool) -> AgentState:
    return {
        "weather_context": WeatherContext(
            status="failed",
            constraints=[_no_address_constraint(english)],
        ),
        "weather_route": "advise_weather",
        "phase": WorkflowPhase.SAFETY,
        "next_action": "advise_weather",
        "audit_log": [
            AuditEvent(
                node_name="address_weather_loader",
                message="Weather lookup skipped because no address context was available.",
            )
        ],
    }


def _resolve_address(address: AddressContext) -> tuple[AddressContext, list[ErrorEvent]]:
    if not address.zip_code or address.status != "pending_zip_lookup":
        return address, []
    try:
        return make_zip_lookup_tool().resolve(address.zip_code), []
    except Exception as exc:
        info = _info_from_exception(exc, "WEATHER", "ZIP")
        error = ErrorEvent.from_info(
            node_name="address_weather_loader",
            info=info,
            recoverable=True,
            surface_to_user=True,
        )
        return address.model_copy(update={"status": "failed"}), [error]


def _unresolved_address_delta(
    address: AddressContext,
    query: str | None,
    errors: list[ErrorEvent],
    english: bool,
) -> AgentState:
    visible_errors = errors or [_unresolved_address_error(address, query)]
    return {
        "address_context": address,
        "weather_context": WeatherContext(
            status="failed",
            location_label=query,
            constraints=[_unresolved_address_constraint(english)],
        ),
        "weather_route": "advise_weather",
        "phase": WorkflowPhase.SAFETY,
        "next_action": "advise_weather",
        "errors": visible_errors,
        "audit_log": [
            AuditEvent(
                node_name="address_weather_loader",
                message="Zip/city resolution did not produce a usable weather query.",
                metadata={"zip_code": address.zip_code, "location_query": query},
            )
        ],
    }


def _load_weather(query: str, english: bool) -> tuple[WeatherContext, list[ErrorEvent]]:
    try:
        return make_weather_tool().get_forecast(query), []
    except Exception as exc:
        info = _info_from_exception(exc, "WEATHER", "FORECAST")
        weather = WeatherContext(
            status="failed",
            location_label=query,
            constraints=[_weather_failure_constraint(english)],
        )
        return weather, [
            ErrorEvent.from_info(
                node_name="address_weather_loader",
                info=info,
                recoverable=True,
                surface_to_user=True,
            )
        ]


def _weather_delta(
    address: AddressContext,
    weather: WeatherContext,
    query: str,
    errors: list[ErrorEvent],
) -> AgentState:
    return {
        "address_context": address,
        "weather_context": weather,
        "weather_route": "advise_weather",
        "phase": WorkflowPhase.SAFETY,
        "next_action": "advise_weather",
        "errors": errors,
        "audit_log": [
            AuditEvent(
                node_name="address_weather_loader",
                message="Loaded weather context.",
                metadata={
                    "status": weather.status,
                    "location_query": query,
                    "source": weather.source,
                },
            )
        ],
    }


def _info_from_exception(exc: Exception, domain: str, step: str):
    if isinstance(exc, BoundaryError):
        return exc.info
    return make_error_info(
        domain=domain,
        step=step,
        category=ErrorCategory.INTERNAL,
        responsibility=f"{step} 执行失败",
        reason=f"{exc.__class__.__name__}: {exc}",
        hint="查看 raw 定位未分类天气链路异常。",
        raw=f"{exc.__class__.__name__}: {exc}",
        retriable=False,
    )


def _unresolved_address_error(address: AddressContext, query: str | None) -> ErrorEvent:
    info = make_error_info(
        domain="WEATHER",
        step="ZIP",
        category=ErrorCategory.NOT_FOUND,
        responsibility="邮编未解析到城市",
        reason=f"邮编 {address.zip_code or query or ''} 无匹配",
        hint="请确认邮编或直接提供城市名。",
        raw=str(address.raw_response or address.model_dump(mode="json")),
        retriable=False,
    )
    return ErrorEvent.from_info(
        node_name="address_weather_loader",
        info=info,
        recoverable=True,
        surface_to_user=True,
    )


def _no_address_constraint(english: bool) -> str:
    return (
        "No city or postal code was provided, so weather could not be checked."
        if english
        else "未提供邮编或城市，无法查询天气。"
    )


def _unresolved_address_constraint(english: bool) -> str:
    return (
        "The postal code could not be resolved to a usable city; weather was not verified."
        if english
        else "邮编未能解析到可用城市，天气未校验。"
    )


def _weather_failure_constraint(english: bool) -> str:
    return (
        "Weather lookup failed; manually verify local weather before formal construction."
        if english
        else "天气查询失败，正式施工前应人工复核当地天气。"
    )
