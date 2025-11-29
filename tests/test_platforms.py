import pytest
from src.platforms import (
    BraintrustPlatform,
    LangSmithPlatform,
    OTLPPlatform,
    ConsolePlatform,
    get_platform
)


def test_braintrust_platform_configuration():
    """Braintrust platform has correct endpoint and headers"""
    platform = BraintrustPlatform(
        api_key="test-key",
        project_name="test-project"
    )
    assert platform.endpoint == "https://api.braintrust.dev/otel/v1/traces"
    headers = platform.get_headers()
    assert "Authorization" in headers
    assert "Bearer test-key" in headers["Authorization"]
    assert "x-bt-parent" in headers


def test_langsmith_platform_configuration():
    """LangSmith platform has correct endpoint and headers"""
    platform = LangSmithPlatform(
        api_key="test-key",
        project_name="test-project"
    )
    # Endpoint should include /v1/traces for OTLPSpanExporter
    assert platform.endpoint == "https://api.smith.langchain.com/otel/v1/traces"
    headers = platform.get_headers()
    assert headers["x-api-key"] == "test-key"
    assert headers["Langsmith-Project"] == "test-project"


def test_otlp_platform_configuration():
    """Generic OTLP platform accepts custom endpoint"""
    platform = OTLPPlatform(
        endpoint_url="http://localhost:4318/v1/traces"
    )
    assert platform.endpoint == "http://localhost:4318/v1/traces"
    assert platform.get_headers() == {}


def test_console_platform_no_export():
    """Console platform is for debugging only"""
    platform = ConsolePlatform()
    assert platform.endpoint is None


def test_get_platform_by_name():
    """get_platform factory creates correct platform type"""
    config = {
        "platform": "braintrust",
        "api_key": "test",
        "project_name": "proj"
    }
    platform = get_platform(config)
    assert isinstance(platform, BraintrustPlatform)
