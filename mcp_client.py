import os
import shutil
import sys
from pathlib import Path
from typing import Any

import airportsdata
import certifi
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

# =========================================================
# Environment setup
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = (
    os.getenv("AVIATION_STACK_API_KEY")
    or os.getenv("AVIATIONSTACK_API_KEY")
)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

WEATHER_SERVER_PATH = BASE_DIR / "custom_weather_mcp_server.py"
UVX_COMMAND = shutil.which("uvx") or "uvx"


def _require_env(name: str, value: str | None) -> str:
    """Return an environment value or raise a readable setup error."""
    if not value:
        raise RuntimeError(
            f"{name} is missing. Add {name}=your_key to the project .env file."
        )
    return value


def _subprocess_env(**updates: str | None) -> dict[str, str]:
    """Preserve current environment and inject MCP API keys."""
    env = os.environ.copy()
    for key, value in updates.items():
        if value:
            env[key] = value
    return env


# =========================================================
# LLM (Lazy initialization)
# =========================================================

_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    """Initialize and cache the ChatGroq model instance."""
    global _llm
    if _llm is None:
        key = _require_env("GROQ_API_KEY", GROQ_API_KEY)
        _llm = ChatGroq(model="openai/gpt-oss-120b", api_key=key)
    return _llm


# =========================================================
# MCP client
# =========================================================

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY or ''}",
        },
        "aviationstack": {
            "transport": "stdio",
            "command": UVX_COMMAND,
            "args": ["aviationstack-mcp"],
            "env": _subprocess_env(
                AVIATION_STACK_API_KEY=AVIATION_STACK_API_KEY,
            ),
        },
        "weather": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(WEATHER_SERVER_PATH)],
            "env": _subprocess_env(
                OPENWEATHER_API_KEY=OPENWEATHER_API_KEY,
            ),
        },
    }
)


async def _get_server_tool(server_name: str, tool_name: str):
    """Load one tool from an MCP server safely."""
    if server_name == "tavily":
        _require_env("TAVILY_API_KEY", TAVILY_API_KEY)

    elif server_name == "aviationstack":
        _require_env("AVIATION_STACK_API_KEY", AVIATION_STACK_API_KEY)
        if shutil.which("uvx") is None:
            raise RuntimeError(
                "uvx was not found. Install uv and ensure it is in PATH."
            )

    elif server_name == "weather":
        _require_env("OPENWEATHER_API_KEY", OPENWEATHER_API_KEY)
        if not WEATHER_SERVER_PATH.is_file():
            raise FileNotFoundError(
                f"Weather MCP server not found at: {WEATHER_SERVER_PATH}"
            )

    tools = await client.get_tools(server_name=server_name)
    tool = next((item for item in tools if item.name == tool_name), None)

    if tool is None:
        available = ", ".join(sorted(item.name for item in tools)) or "none"
        raise RuntimeError(
            f"Tool '{tool_name}' not found on server '{server_name}'. Available: {available}"
        )

    return tool


# =========================================================
# Tool Execution Wrappers
# =========================================================

async def tavily_mcp_search(query: str):
    search_tool = await _get_server_tool("tavily", "tavily_search")
    return await search_tool.ainvoke({"query": query})


async def aviation_mcp_call(tool_name: str, tool_args: dict[str, Any] | None = None):
    aviation_tool = await _get_server_tool("aviationstack", tool_name)
    return await aviation_tool.ainvoke(tool_args or {})


async def weather_mcp_search(city: str):
    weather_tool = await _get_server_tool("weather", "get_current_weather")
    return await weather_tool.ainvoke({"city": city})


async def forecast_mcp_search(city: str):
    forecast_tool = await _get_server_tool("weather", "get_forecast")
    return await forecast_tool.ainvoke({"city": city})


# =========================================================
# Helpers: Extraction & Airport Lookup
# =========================================================

def extract_destination(query: str) -> str:
    """Extract destination name from natural language query."""
    prompt = f"""Extract only the destination city or country from the travel request.
Travel request: {query}
Return only the destination name without punctuation or markdown formatting."""

    response = get_llm().invoke(prompt)
    destination = str(response.content).strip().strip("\"' .")

    if not destination:
        raise ValueError("The destination could not be extracted.")

    return destination


def find_airports_by_city(city_name: str) -> list[dict[str, Any]]:
    """Look up IATA airport codes for a city using airportsdata."""
    airports = airportsdata.load("IATA")
    query_lower = city_name.strip().lower()

    matches = [
        {"iata": code, "name": meta["name"], "country": meta["country"]}
        for code, meta in airports.items()
        if meta.get("city", "").lower() == query_lower
    ]
    return matches