import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

mcp = FastMCP("Weather MCP Server")

REQUEST_TIMEOUT_SECONDS = 20


def _get_api_key() -> str:
    """Read API key dynamically from environment."""
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENWEATHER_API_KEY is missing from environment or project .env file."
        )
    return key


def _request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Execute OpenWeather API request with friendly error formatting."""
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code == 404:
            return {
                "error": True,
                "message": f"Location not found. Please verify the city name.",
            }

        response.raise_for_status()
        return response.json()

    except requests.RequestException as exc:
        details = ""
        failed_response = getattr(exc, "response", None)
        if failed_response is not None:
            details = f" Response: {failed_response.text[:300]}"

        return {
            "error": True,
            "message": f"OpenWeather request failed: {exc}.{details}",
        }


@mcp.tool()
def get_current_weather(city: str, country_code: str | None = None) -> dict[str, Any]:
    """Return the current weather for a specific city and optional ISO country code."""
    query = city.strip()
    if not query:
        return {"error": True, "message": "City name cannot be empty."}

    if country_code:
        query = f"{query},{country_code.strip()}"

    data = _request_json(
        "https://api.openweathermap.org/data/2.5/weather",
        {
            "q": query,
            "appid": _get_api_key(),
            "units": "metric",
        },
    )

    if data.get("error"):
        return data

    return {
        "city": data.get("name", city),
        "country": data.get("sys", {}).get("country", ""),
        "temperature_c": data.get("main", {}).get("temp"),
        "feels_like_c": data.get("main", {}).get("feels_like"),
        "humidity_pct": data.get("main", {}).get("humidity"),
        "condition": (
            data.get("weather", [{}])[0].get("description", "Unknown")
        ),
        "wind_speed_mps": data.get("wind", {}).get("speed"),
    }


@mcp.tool()
def get_forecast(city: str, country_code: str | None = None) -> dict[str, Any]:
    """Return the 5 upcoming 3-hour forecast intervals for a city."""
    query = city.strip()
    if not query:
        return {"error": True, "message": "City name cannot be empty."}

    if country_code:
        query = f"{query},{country_code.strip()}"

    data = _request_json(
        "https://api.openweathermap.org/data/2.5/forecast",
        {
            "q": query,
            "appid": _get_api_key(),
            "units": "metric",
        },
    )

    if data.get("error"):
        return data

    forecast_items = [
        {
            "datetime": item.get("dt_txt"),
            "temperature_c": item.get("main", {}).get("temp"),
            "condition": (
                item.get("weather", [{}])[0].get("description", "Unknown")
            ),
        }
        for item in data.get("list", [])[:5]
    ]

    return {
        "city": data.get("city", {}).get("name", city),
        "country": data.get("city", {}).get("country", ""),
        "forecast": forecast_items,
    }


if __name__ == "__main__":
    # Runs the FastMCP server over standard input/output
    mcp.run(transport="stdio")