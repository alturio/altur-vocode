"""
Parameter formatters for external action requests.

This module contains formatters that transform parameter values based on format specifications
in the action's input schema (x-formats field). This allows for automatic conversion of
parameter values to formats expected by external APIs.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Union
import pytz
from loguru import logger


def convert_datetime_to_epoch(
    datetime_str: str, format_type: str, timezone_str: Optional[str] = None
) -> Union[int, str]:
    """
    Convert ISO 8601 datetime string to epoch format.

    Args:
        datetime_str: ISO 8601 datetime string (e.g., "2025-09-06T10:00:00-05:00",
                     "2025-09-06T15:00:00Z", or "2025-09-06T10:00:00")
        format_type: "epoch_s" for seconds or "epoch_ms" for milliseconds
        timezone_str: Timezone string (e.g., "America/Mexico_City") for naive datetimes.
                     Defaults to UTC if not provided.

    Returns:
        int: Epoch timestamp in seconds or milliseconds
        str: Original datetime string if conversion fails

    Examples:
        >>> convert_datetime_to_epoch("2025-09-06T10:00:00-05:00", "epoch_s")
        1757088000
        >>> convert_datetime_to_epoch("2025-09-06T15:00:00Z", "epoch_ms")
        1757088000000
        >>> convert_datetime_to_epoch("2025-09-06T10:00:00", "epoch_s", "America/Mexico_City")
        1757088000
    """
    try:
        # Handle 'Z' notation for UTC
        datetime_str_normalized = datetime_str.replace("Z", "+00:00")

        # Parse ISO 8601 datetime string
        dt = datetime.fromisoformat(datetime_str_normalized)

        # If naive (no timezone info), localize to provided timezone or UTC
        if dt.tzinfo is None:
            if timezone_str:
                try:
                    tz = pytz.timezone(timezone_str)
                    dt = tz.localize(dt)
                except pytz.UnknownTimeZoneError:
                    logger.warning(
                        f"Unknown timezone '{timezone_str}', defaulting to UTC for datetime conversion"
                    )
                    dt = pytz.UTC.localize(dt)
            else:
                dt = pytz.UTC.localize(dt)

        # Convert to epoch
        epoch_seconds = dt.timestamp()

        # Return in requested format
        if format_type == "epoch_s":
            return int(epoch_seconds)
        elif format_type == "epoch_ms":
            return int(epoch_seconds * 1000)
        else:
            logger.warning(
                f"Unknown format type '{format_type}', keeping original value"
            )
            return datetime_str

    except (ValueError, AttributeError) as e:
        logger.warning(
            f"Failed to convert datetime '{datetime_str}' to epoch: {e}. Keeping original value."
        )
        return datetime_str


def apply_parameter_format(
    param_value: Any, format_type: str, extra_context: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Apply formatting to a parameter value based on the format type.

    This is the main entry point for parameter formatting. It delegates to specific
    formatters based on the format_type.

    Args:
        param_value: The parameter value to format
        format_type: The format type (e.g., "epoch_s", "epoch_ms")
        extra_context: Optional extra context dict (e.g., {"timezone": "America/Mexico_City"})

    Returns:
        The formatted value, or the original value if formatting fails or is not applicable

    Examples:
        >>> apply_parameter_format("2025-09-06T10:00:00Z", "epoch_s")
        1757088000
        >>> apply_parameter_format("some-value", "unknown_format")
        "some-value"
    """
    if format_type in ["epoch_s", "epoch_ms"] and isinstance(param_value, str):
        timezone_str = None
        if extra_context:
            timezone_str = extra_context.get("timezone")
        return convert_datetime_to_epoch(param_value, format_type, timezone_str)

    return param_value


def apply_parameter_formats(
    payload: Dict[str, Any],
    param_formats: Dict[str, str],
    extra_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Apply formatting to multiple parameters in a payload based on format specifications.

    Args:
        payload: Dictionary of parameter names to values
        param_formats: Dictionary mapping parameter names to format types
        extra_context: Optional extra context dict (e.g., {"timezone": "America/Mexico_City"})

    Returns:
        New dictionary with formatted parameter values

    Examples:
        >>> payload = {"date": "2025-09-06T10:00:00Z", "name": "John"}
        >>> formats = {"date": "epoch_s"}
        >>> apply_parameter_formats(payload, formats)
        {"date": 1757088000, "name": "John"}
    """
    if not param_formats:
        return payload

    formatted_payload = payload.copy()

    for param_name, param_value in payload.items():
        if param_name in param_formats:
            format_type = param_formats[param_name]
            formatted_value = apply_parameter_format(
                param_value, format_type, extra_context
            )
            formatted_payload[param_name] = formatted_value

    return formatted_payload
