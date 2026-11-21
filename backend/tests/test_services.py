import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from typing import TypedDict, TypeVar
from unittest.mock import AsyncMock, patch

import httpx
from google import genai
from google.genai import types

from app import services


T = TypeVar("T")


class TavilyRequestBody(TypedDict):
    query: str
    search_depth: str
    max_results: int
    topic: str
    include_answer: bool
    include_raw_content: bool
    include_published_date: bool
    include_usage: bool


class CapturedHTTPCall(TypedDict):
    headers: dict[str, str]
    json: TavilyRequestBody


class GeminiSearchRequest(TypedDict):
    model: str
    input: str
    tools: list[dict[str, str]]
    store: bool


class GeminiImageRequest(TypedDict):
    model: str
    contents: list[str | types.Part]
    config: types.GenerateContentConfig


class GeminiMediaRequest(TypedDict):
    model: str
    contents: list[str | types.Part]


class GeminiImageClientOptions(TypedDict):
    api_key: str
    http_options: types.HttpOptions


def required(value: T | None, description: str) -> T:
    if value is None:
        raise AssertionError(f"Expected {description} to be set by the fake provider")
    return value


def tavily_body(call: CapturedHTTPCall) -> TavilyRequestBody:
    return call["json"]


def citation(url="https://example.org/report", title="Example report", start=0, end=12):
    return SimpleNamespace(
        type="url_citation",
        url=url,
        title=title,
        start_index=start,
        end_index=end,
    )


def interaction_with_citations(*annotations, text="Evidence text."):
    content = SimpleNamespace(type="text", text=text, annotations=list(annotations))
    step = SimpleNamespace(type="model_output", content=[content])
    return SimpleNamespace(steps=[step])


def gemini_image_payload(
    assessment="supported",
    likelihood="low",
    confidence=0.82,
    visual=None,
    indicators=None,
    inconclusive=None,
):
    return {
        "assessment": assessment,
        "summary": "Visible details are broadly consistent with the assessment, but this is not certification.",
        "visual_observations": visual or ["The main subject has consistent edges and lighting."],
        "suspicious_indicators": indicators or [],
        "manipulation_likelihood": likelihood,
        "model_confidence": confidence,
        "metadata_observations": ["No selected embedded EXIF fields were supplied."],
        "inconclusive_reasons": inconclusive or [],
        "limitations": ["A single visual review cannot establish image provenance."],
        "recommended_checks": ["Compare with the original file and an independent source."],
    }


class FakeHTTPResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.is_error = status_code >= 400

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class FakeAsyncHTTPClient:
    calls: list[tuple[str, CapturedHTTPCall]] = []
    payload: object = None
    status_code: int = 200
    created: int = 0
    error: Exception | None = None

    def __init__(self, **kwargs: object) -> None:
        self.created += 1
        self.kwargs = kwargs

    async def __aenter__(self) -> "FakeAsyncHTTPClient":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: TavilyRequestBody,
    ) -> FakeHTTPResponse:
        self.calls.append((url, {"headers": headers, "json": json}))
        error = self.error
        if error is not None:
            raise error
        return FakeHTTPResponse(self.payload, self.status_code)


class ServicesTests(unittest.TestCase):
    def test_gemini_interaction_uses_configured_model_google_search_and_maps_citations(self):
        response = interaction_with_citations(citation(start=0, end=8), text="Evidence text.")

        class FakeGeminiClient:
            request: GeminiSearchRequest | None = None
            closed: bool = False

            def __init__(self, api_key: str) -> None:
                self.api_key = api_key
                self.interactions = self

            def create(
                self,
                *,
                model: str,
                input: str,
                tools: list[dict[str, str]],
                store: bool,
            ) -> SimpleNamespace:
                FakeGeminiClient.request = {"model": model, "input": input, "tools": tools, "store": store}
                return response

            def close(self) -> None:
                FakeGeminiClient.closed = True

        with patch.object(genai, "Client", FakeGeminiClient):
            sources = services._gemini_search_sync("A factual claim", "test-token")

        self.assertEqual(services.GEMINI_MODEL, "gemini-3.5-flash-lite")
        request = required(FakeGeminiClient.request, "Gemini interaction request")
        self.assertEqual(request["model"], services.GEMINI_MODEL)
        self.assertEqual(request["tools"], [{"type": "google_search"}])
        self.assertFalse(request["store"])
        self.assertIn("A factual claim", request["input"])
        self.assertTrue(FakeGeminiClient.closed)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["url"], "https://example.org/report")
        self.assertEqual(sources[0]["publisher"], "example.org")
        self.assertEqual(sources[0]["snippet"], "Evidence")

    def test_citation_parser_ignores_non_http_and_malformed_annotations(self):
        result = interaction_with_citations(
            citation(url="javascript:alert(1)", start=0, end=8),
            citation(url="https://example.org/bad-range", start=50, end=80),
            SimpleNamespace(type="other", url="https://example.org/ignored"),
        )

        sources = services._sources_from_gemini_interaction(result)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["url"], "https://example.org/bad-range")
        self.assertEqual(sources[0]["snippet"], "")

    def test_tavily_search_uses_basic_five_result_request_and_maps_metadata(self):
        FakeAsyncHTTPClient.calls = []
        FakeAsyncHTTPClient.payload = {"results": [{
            "title": "Example report", "url": "https://www.example.org/report",
            "content": "A source snippet", "published_date": "2026-01-15",
        }], "usage": {"credits": 1}}
        FakeAsyncHTTPClient.status_code = 200
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-token"}), patch.object(
            services.httpx, "AsyncClient", FakeAsyncHTTPClient
        ):
            sources = asyncio.run(services.search_sources("A factual claim"))

        url, request = FakeAsyncHTTPClient.calls[0]
        body = tavily_body(request)
        self.assertEqual(url, "https://api.tavily.com/search")
        self.assertEqual(request["headers"]["Authorization"], "Bearer test-token")
        self.assertEqual(body["max_results"], 5)
        self.assertEqual(body["search_depth"], "basic")
        self.assertFalse(body["include_answer"])
        self.assertEqual(sources, [{
            "title": "Example report", "publisher": "example.org",
            "url": "https://www.example.org/report", "snippet": "A source snippet",
            "published_at": "2026-01-15", "provider": "Tavily",
        }])

    def test_tavily_success_extracts_five_results(self):
        FakeAsyncHTTPClient.calls = []
        FakeAsyncHTTPClient.error = None
        FakeAsyncHTTPClient.status_code = 200
        FakeAsyncHTTPClient.payload = {"results": [
            {"title": f"Source {i}", "url": f"https://example.org/{i}", "content": f"Evidence {i}"}
            for i in range(5)
        ]}
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-token"}), patch.object(
            services.httpx, "AsyncClient", FakeAsyncHTTPClient
        ):
            sources = asyncio.run(services.search_sources("A factual claim"))
        self.assertEqual(len(sources), 5)
        self.assertEqual(sources[4]["snippet"], "Evidence 4")

    def test_tavily_caps_results_and_ignores_invalid_urls_and_missing_fields(self):
        FakeAsyncHTTPClient.calls = []
        FakeAsyncHTTPClient.status_code = 200
        FakeAsyncHTTPClient.payload = {"results": [
            {"url": "javascript:alert(1)", "title": "unsafe"},
            *[{"url": f"https://example.org/{i}", "content": "Actual returned snippet"} for i in range(7)],
        ]}
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-token"}), patch.object(
            services.httpx, "AsyncClient", FakeAsyncHTTPClient
        ):
            sources = asyncio.run(services.search_sources("A factual claim"))

        self.assertEqual(len(sources), 5)
        self.assertTrue(all(s["title"] == "example.org" and s["snippet"] == "Actual returned snippet" for s in sources))
        self.assertTrue(all(s["published_at"] is None for s in sources))

    def test_tavily_sources_without_snippets_are_not_used_as_evidence(self):
        FakeAsyncHTTPClient.status_code = 200
        FakeAsyncHTTPClient.payload = {"results": [{"title": "Title", "url": "https://example.org"}]}
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-token"}), patch.object(
            services.httpx, "AsyncClient", FakeAsyncHTTPClient
        ):
            with self.assertRaisesRegex(services.EvidenceRetrievalError, "could not be parsed"):
                asyncio.run(services.search_sources("A factual claim"))

    def test_missing_tavily_key_and_provider_errors_fail_closed(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}):
            with self.assertRaises(services.EvidenceRetrievalError):
                asyncio.run(services.search_sources("A factual claim"))

        for status in (401, 429, 432, 433, 500):
            FakeAsyncHTTPClient.error = None
            FakeAsyncHTTPClient.status_code = status
            FakeAsyncHTTPClient.calls = []
            with patch.dict(os.environ, {"TAVILY_API_KEY": "test-token"}), patch.object(
                services.httpx, "AsyncClient", FakeAsyncHTTPClient
            ):
                with self.assertRaises(services.EvidenceRetrievalError) as raised:
                    asyncio.run(services.search_sources("A factual claim"))
            self.assertNotIn("test-token", str(raised.exception))

    def test_tavily_timeout_is_reported_without_sensitive_data(self):
        FakeAsyncHTTPClient.error = services.httpx.TimeoutException("request timeout")
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-token"}), patch.object(
            services.httpx, "AsyncClient", FakeAsyncHTTPClient
        ):
            with self.assertRaisesRegex(services.EvidenceRetrievalError, "timed out"):
                asyncio.run(services.search_sources("A factual claim"))
        FakeAsyncHTTPClient.error = None

    def test_malformed_tavily_response_and_unusable_sources_are_reported(self):
        FakeAsyncHTTPClient.status_code = 200
        for payload in ({"not_results": []}, {"results": "not-an-array"}, {"results": [{"url": "javascript:alert(1)", "content": "bad"}]}):
            FakeAsyncHTTPClient.payload = payload
            with patch.dict(os.environ, {"TAVILY_API_KEY": "test-token"}), patch.object(
                services.httpx, "AsyncClient", FakeAsyncHTTPClient
            ):
                with self.assertRaises(services.EvidenceRetrievalError):
                    asyncio.run(services.search_sources("A factual claim"))

    def test_malformed_tavily_json_is_reported(self):
        class InvalidJSONResponse(FakeHTTPResponse):
            def json(self) -> object:
                raise ValueError("not JSON")

        class InvalidJSONClient(FakeAsyncHTTPClient):
            async def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: TavilyRequestBody,
            ) -> FakeHTTPResponse:
                self.calls.append((url, {"headers": headers, "json": json}))
                return InvalidJSONResponse(None, 200)

        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-token"}), patch.object(
            services.httpx, "AsyncClient", InvalidJSONClient
        ):
            with self.assertRaises(services.EvidenceRetrievalError):
                asyncio.run(services.search_sources("A factual claim"))

    def test_empty_tavily_results_return_empty_without_invented_sources(self):
        FakeAsyncHTTPClient.status_code = 200
        FakeAsyncHTTPClient.payload = {"results": []}
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-token"}), patch.object(
            services.httpx, "AsyncClient", FakeAsyncHTTPClient
        ):
            self.assertEqual(asyncio.run(services.search_sources("A factual claim")), [])

    def test_grounding_helper_remains_unchanged_and_not_used_for_active_retrieval(self):
        self.assertTrue(callable(services.search_sources_gemini))
        self.assertTrue(callable(services._sources_from_gemini_interaction))

    def test_no_sources_returns_unavailable_without_a_gemini_request(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-token"}):
            result = asyncio.run(services.assess_claim("A factual claim", []))
        self.assertEqual(result["verdict"], "insufficient_evidence")
        self.assertIn("Evidence retrieval is unavailable", result["explanation"])

    def test_missing_gemini_key_returns_review_unavailable(self):
        source = {"title": "Source", "url": "https://example.org", "snippet": "Text"}
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            result = asyncio.run(services.assess_claim("A factual claim", [source]))
        self.assertEqual(result["verdict"], "review_unavailable")
        self.assertIn("GEMINI_API_KEY", result["limitations"][0])

    def test_gemini_verdict_uses_configured_model_structured_schema_and_source_metadata(self):
        source = {"title":"Example report", "publisher":"Example News", "url":"https://example.org/report", "published_at":"2025-03-01", "snippet":"Evidence text."}
        sources = [source, *[
            {"title":f"Report {index}", "publisher":f"Publisher {index}", "url":f"https://example.org/{index}", "published_at":None, "snippet":f"Evidence snippet {index}."}
            for index in range(2, 6)
        ]]

        class FakeGeminiClient:
            request = None
            closed = False
            def __init__(self, *, api_key, http_options):
                FakeGeminiClient.http_options = http_options
                self.models = self
            def generate_content(self, *, model, contents, config):
                FakeGeminiClient.request = {"model":model, "contents":contents, "config":config}
                payload = {"verdict":"TRUE", "summary":"The supplied evidence supports the claim.", "model_confidence":0.88, "evidence_coverage":"Two independent sources directly address the claim.", "findings":["Both sources agree."], "supporting_evidence":["Evidence text."], "contradicting_evidence":[], "limitations":[]}
                return SimpleNamespace(text=json.dumps(payload))
            def close(self):
                FakeGeminiClient.closed = True

        with patch.dict(os.environ, {"GEMINI_API_KEY":"test-token", "GEMINI_MODEL":"test-gemini-model"}), patch.object(genai, "Client", FakeGeminiClient):
            result = asyncio.run(services.assess_claim("A factual claim", sources))
        self.assertEqual(result["verdict"], "supported")
        self.assertEqual(result["model_confidence"], 0.88)
        self.assertTrue(FakeGeminiClient.closed)
        request = required(FakeGeminiClient.request, "Gemini verdict request")
        self.assertEqual(request["model"], "test-gemini-model")
        self.assertEqual(FakeGeminiClient.http_options.timeout, services.GEMINI_VERDICT_TIMEOUT_MS)
        self.assertEqual(request["config"].response_mime_type, "application/json")
        self.assertIs(request["config"].response_schema, services.GeminiVerdictReview)
        self.assertNotIn("additionalProperties", services.GeminiVerdictReview.model_json_schema())
        self.assertIn('"claim": "A factual claim"', request["contents"])
        for supplied in sources:
            for field in ("title", "publisher", "url", "snippet"):
                self.assertIn(supplied[field], request["contents"])

    def test_gemini_verdict_failure_returns_review_unavailable_and_logs_safely(self):
        source = {"title":"Example", "url":"https://example.org", "snippet":"Evidence"}
        class FailingGeminiClient:
            def __init__(self, *, api_key, http_options): self.models = self
            def generate_content(self, **kwargs): raise RuntimeError("provider unavailable for token dummy-gemini-key")
            def close(self): pass
        with patch.dict(os.environ, {"GEMINI_API_KEY":"dummy-gemini-key"}), patch.object(genai, "Client", FailingGeminiClient), self.assertLogs(services.logger, level="ERROR") as logs:
            result = asyncio.run(services.assess_claim("A factual claim", [source]))
        self.assertEqual(result["verdict"], "review_unavailable")
        self.assertIsNone(result["model_confidence"])
        joined_logs = "\n".join(logs.output)
        self.assertIn("provider unavailable", joined_logs)
        self.assertNotIn("dummy-gemini-key", joined_logs)

    def test_gemini_auth_and_quota_errors_fail_closed_with_clear_limitations(self):
        source = {"title":"Example", "url":"https://example.org", "snippet":"Evidence"}
        for status, expected in ((401, "authentication_or_permission"), (429, "quota_or_rate_limit")):
            request = httpx.Request("POST", "https://generativelanguage.googleapis.com")
            failure = httpx.HTTPStatusError("provider error", request=request, response=httpx.Response(status, request=request))
            class FailingGeminiClient:
                attempts = 0
                def __init__(self, *, api_key, http_options): self.models = self
                def generate_content(self, **kwargs):
                    FailingGeminiClient.attempts += 1
                    raise failure
                def close(self): pass
            with self.subTest(status=status), patch.dict(os.environ, {"GEMINI_API_KEY":"test-token"}), patch.object(genai, "Client", FailingGeminiClient):
                with patch.object(services.time, "sleep"):
                    result = asyncio.run(services.assess_claim("A factual claim", [source]))
            self.assertEqual(result["verdict"], "review_unavailable")
            self.assertIn(expected, result["limitations"][0])
            if status == 429:
                self.assertEqual(FailingGeminiClient.attempts, services.GEMINI_VERDICT_MAX_ATTEMPTS)

    def test_invalid_gemini_response_fails_validation(self):
        source = {"title":"Example", "url":"https://example.org", "snippet":"Evidence"}
        class InvalidGeminiClient:
            def __init__(self, *, api_key, http_options): self.models = self
            def generate_content(self, **kwargs): return SimpleNamespace(text='{"verdict":"CERTAINLY_TRUE"}')
            def close(self): pass
        with patch.dict(os.environ, {"GEMINI_API_KEY":"test-token"}), patch.object(genai, "Client", InvalidGeminiClient):
            result = asyncio.run(services.assess_claim("A factual claim", [source]))
        self.assertEqual(result["verdict"], "review_unavailable")
        self.assertIsNone(result["model_confidence"])

    def test_malformed_gemini_json_fails_closed(self):
        source = {"title":"Example", "url":"https://example.org", "snippet":"Evidence"}
        class MalformedGeminiClient:
            def __init__(self, *, api_key, http_options): self.models = self
            def generate_content(self, **kwargs): return SimpleNamespace(text='{"verdict":')
            def close(self): pass
        with patch.dict(os.environ, {"GEMINI_API_KEY":"test-token"}), patch.object(genai, "Client", MalformedGeminiClient):
            result = asyncio.run(services.assess_claim("A factual claim", [source]))
        self.assertEqual(result["verdict"], "review_unavailable")
        self.assertIsNone(result["model_confidence"])

    def test_unavailable_gemini_model_is_logged_and_does_not_return_a_verdict(self):
        source = {"title":"Example", "url":"https://example.org", "snippet":"Evidence"}
        request = httpx.Request("POST", "https://generativelanguage.googleapis.com")
        failure = httpx.HTTPStatusError("model not found", request=request, response=httpx.Response(404, request=request))
        class MissingModelClient:
            def __init__(self, *, api_key, http_options): self.models = self
            def generate_content(self, **kwargs): raise failure
            def close(self): pass
        with patch.dict(os.environ, {"GEMINI_API_KEY":"test-token", "GEMINI_MODEL":"missing-model"}), patch.object(genai, "Client", MissingModelClient), self.assertLogs(services.logger, level="ERROR") as logs:
            result = asyncio.run(services.assess_claim("A factual claim", [source]))
        self.assertEqual(result["verdict"], "review_unavailable")
        self.assertIn("invalid_model_or_endpoint", result["limitations"][0])
        self.assertIn("provider_status=404", "\n".join(logs.output))

    def test_conflicting_support_and_contradiction_returns_insufficient_evidence(self):
        source = {"title":"Example", "url":"https://example.org", "snippet":"Evidence"}
        class ConflictingGeminiClient:
            def __init__(self, *, api_key, http_options): self.models = self
            def generate_content(self, **kwargs):
                payload = {"verdict":"TRUE", "summary":"Sources disagree.", "model_confidence":0.9, "evidence_coverage":"Conflicting", "findings":[], "supporting_evidence":["One source supports the claim."], "contradicting_evidence":["Another source contradicts it."], "limitations":[]}
                return SimpleNamespace(text=json.dumps(payload))
            def close(self): pass
        with patch.dict(os.environ, {"GEMINI_API_KEY":"test-token"}), patch.object(genai, "Client", ConflictingGeminiClient):
            result = asyncio.run(services.assess_claim("A factual claim", [source]))
            self.assertEqual(result["verdict"], "insufficient_evidence")
        self.assertLessEqual(result["model_confidence"], 0.4)

    def test_gemini_timeout_returns_unavailable_without_claiming_insufficient_evidence(self):
        source = {"title":"Example", "url":"https://example.org", "snippet":"Evidence"}
        class TimedOutGeminiClient:
            def __init__(self, *, api_key, http_options): self.models = self
            def generate_content(self, **kwargs): raise services.httpx.ReadTimeout("timed out")
            def close(self): pass
        with patch.dict(os.environ, {"GEMINI_API_KEY":"test-token"}), patch.object(genai, "Client", TimedOutGeminiClient):
            result = asyncio.run(services.assess_claim("A factual claim", [source]))
        self.assertEqual(result["verdict"], "review_unavailable")
        self.assertIsNone(result["model_confidence"])
        self.assertIn("network_timeout", result["limitations"][0])

    def test_gemini_transient_server_error_retries_and_then_succeeds(self):
        source = {"title":"Example", "url":"https://example.org", "snippet":"Evidence"}
        payload = {"verdict":"TRUE", "summary":"Supported by the source.", "model_confidence":0.8, "evidence_coverage":"Direct evidence.", "findings":[], "supporting_evidence":["Evidence."], "contradicting_evidence":[], "limitations":[]}
        class RetryingGeminiClient:
            attempts = 0
            def __init__(self, *, api_key, http_options): self.models = self
            def generate_content(self, **kwargs):
                RetryingGeminiClient.attempts += 1
                if RetryingGeminiClient.attempts == 1:
                    request = httpx.Request("POST", "https://generativelanguage.googleapis.com")
                    raise httpx.HTTPStatusError("busy", request=request, response=httpx.Response(503, request=request))
                return SimpleNamespace(text=json.dumps(payload))
            def close(self): pass
        with patch.dict(os.environ, {"GEMINI_API_KEY":"test-token"}), patch.object(genai, "Client", RetryingGeminiClient), patch.object(services.time, "sleep"):
            result = asyncio.run(services.assess_claim("A factual claim", [source]))
        self.assertEqual(result["verdict"], "supported")
        self.assertEqual(RetryingGeminiClient.attempts, 2)

    def test_gemini_image_analysis_uses_structured_response_and_image_bytes(self):
        class FakeGeminiClient:
            request: GeminiImageRequest | None = None
            init_kwargs: GeminiImageClientOptions | None = None
            closed: bool = False

            def __init__(self, *, api_key: str, http_options: types.HttpOptions) -> None:
                FakeGeminiClient.init_kwargs = {"api_key": api_key, "http_options": http_options}
                self.models = self

            def generate_content(
                self,
                *,
                model: str,
                contents: list[str | types.Part],
                config: types.GenerateContentConfig,
            ) -> SimpleNamespace:
                FakeGeminiClient.request = {"model": model, "contents": contents, "config": config}
                return SimpleNamespace(text=json.dumps(gemini_image_payload(
                    assessment="suspicious",
                    likelihood="high",
                    indicators=["The subject boundary has a sharp brightness discontinuity.", "Foreground and background shadows point in conflicting directions."],
                    visual=["The subject boundary has a sharp brightness discontinuity.", "Foreground and background shadows point in conflicting directions."],
                )))

            def close(self) -> None:
                FakeGeminiClient.closed = True

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key", "GEMINI_IMAGE_MODEL": "gemini-3.8-flash"}), patch.object(
            genai, "Client", FakeGeminiClient
        ):
            result = asyncio.run(services.analyze_image_gemini(b"image bytes", "image/png", {"Software": "Camera"}))

        self.assertEqual(result["assessment"], "suspicious")
        self.assertEqual(result["manipulation_likelihood"], "high")
        self.assertEqual(result["model_confidence"], 0.82)
        self.assertEqual(len(result["suspicious_indicators"]), 2)
        self.assertEqual(result["metadata_observations"], ["No selected embedded EXIF fields were supplied."])
        request = required(FakeGeminiClient.request, "Gemini image request")
        self.assertEqual(request["model"], "gemini-3.8-flash")
        image_part = request["contents"][1]
        if not isinstance(image_part, types.Part):
            raise AssertionError("Expected the Gemini image part")
        inline_data = required(image_part.inline_data, "inline Gemini image data")
        self.assertEqual(inline_data.data, b"image bytes")
        self.assertEqual(inline_data.mime_type, "image/png")
        prompt = request["contents"][0]
        if not isinstance(prompt, str):
            raise AssertionError("Expected the Gemini analysis prompt")
        self.assertIn('"Software": "Camera"', prompt)
        config = request["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertIs(config.response_schema, services.GeminiImageForensics)
        self.assertNotIn("additionalProperties", services.GeminiImageForensics.model_json_schema())
        init_kwargs = required(FakeGeminiClient.init_kwargs, "Gemini client options")
        self.assertEqual(init_kwargs["http_options"].timeout, services.GEMINI_IMAGE_TIMEOUT_MS)
        self.assertEqual(services.GEMINI_IMAGE_TIMEOUT_MS, 90_000)
        self.assertTrue(FakeGeminiClient.closed)

    def test_large_gemini_image_uses_temporary_files_api_upload_and_deletes_file(self):
        class FakeFiles:
            uploaded_content: bytes | None = None
            upload_config: types.UploadFileConfig | None = None
            deleted_name: str | None = None

            def upload(self, *, file, config: types.UploadFileConfig) -> SimpleNamespace:
                FakeFiles.uploaded_content = file.read()
                FakeFiles.upload_config = config
                return SimpleNamespace(uri="https://files.example/image", name="files/temp-image")

            def delete(self, *, name: str) -> None:
                FakeFiles.deleted_name = name

        class FakeGeminiClient:
            request: GeminiImageRequest | None = None

            def __init__(self, **kwargs: object) -> None:
                self.models = self
                self.files = FakeFiles()

            def generate_content(
                self,
                *,
                model: str,
                contents: list[str | types.Part],
                config: types.GenerateContentConfig,
            ) -> SimpleNamespace:
                FakeGeminiClient.request = {"model": model, "contents": contents, "config": config}
                return SimpleNamespace(text=json.dumps(gemini_image_payload(
                    assessment="inconclusive",
                    likelihood="unavailable",
                    confidence=None,
                    visual=["Only a small, blurred region is available."],
                    inconclusive=["Image detail is insufficient for a reliable assessment."],
                )))

            def close(self) -> None: pass

        image_bytes = b"large-test-image"
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"}), patch.object(
            services, "GEMINI_INLINE_IMAGE_MAX_BYTES", 4
        ), patch.object(genai, "Client", FakeGeminiClient):
            asyncio.run(services.analyze_image_gemini(image_bytes, "image/webp", {}))

        uploaded_content = required(FakeFiles.uploaded_content, "uploaded image bytes")
        upload_config = required(FakeFiles.upload_config, "Gemini upload config")
        deleted_name = required(FakeFiles.deleted_name, "temporary file deletion")
        request = required(FakeGeminiClient.request, "Gemini Files API request")
        self.assertEqual(uploaded_content, image_bytes)
        self.assertEqual(upload_config.mime_type, "image/webp")
        self.assertEqual(deleted_name, "files/temp-image")
        image_part = request["contents"][1]
        if not isinstance(image_part, types.Part):
            raise AssertionError("Expected a Gemini file image part")
        file_data = required(image_part.file_data, "Gemini image file data")
        self.assertEqual(file_data.file_uri, "https://files.example/image")
        self.assertEqual(file_data.mime_type, "image/webp")

    def test_gemini_image_analysis_rejects_malformed_structured_response(self):
        class FakeGeminiClient:
            def __init__(self, **kwargs): self.models = self
            def generate_content(self, **kwargs):
                return SimpleNamespace(text='{"visual_observations": "not-an-array"}')
            def close(self): pass

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"}), patch.object(genai, "Client", FakeGeminiClient):
            with self.assertRaises(ValueError):
                asyncio.run(services.analyze_image_gemini(b"image", "image/jpeg", {}))

    def test_gemini_image_analysis_logs_provider_error_without_api_key(self):
        class ProviderError(Exception):
            code = 503

        class FakeGeminiClient:
            def __init__(self, **kwargs): self.models = self
            def generate_content(self, **kwargs): raise ProviderError("provider rejected key test-gemini-secret")
            def close(self): pass

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-secret", "GEMINI_IMAGE_MODEL": "gemini-3.8-flash"}), patch.object(
            genai, "Client", FakeGeminiClient
        ), patch.object(services.asyncio, "sleep", new=AsyncMock()), self.assertLogs(services.logger, level="ERROR") as logs:
            with self.assertRaises(ProviderError):
                asyncio.run(services.analyze_image_gemini(b"image", "image/jpeg", {}))
        joined = "\n".join(logs.output)
        self.assertIn("failure_category=provider_server_error", joined)
        self.assertIn("error_type=ProviderError", joined)
        self.assertIn("provider_status=503", joined)
        self.assertIn("provider_message=provider rejected key [REDACTED]", joined)
        self.assertNotIn("test-gemini-secret", joined)

    def test_gemini_image_analysis_retries_temporary_provider_errors_with_backoff(self):
        class ProviderError(Exception):
            code = 429

        class FakeGeminiClient:
            calls: int = 0

            def __init__(self, **kwargs):
                self.models = self

            def generate_content(self, **kwargs):
                FakeGeminiClient.calls += 1
                if FakeGeminiClient.calls < 3:
                    raise ProviderError("temporary rate limit")
                return SimpleNamespace(text=json.dumps(gemini_image_payload(
                    assessment="suspicious",
                    likelihood="medium",
                    indicators=["A visible seam appears near the object.", "Texture changes abruptly on one side of the seam."],
                    visual=["A visible seam appears near the object.", "Texture changes abruptly on one side of the seam."],
                )))

            def close(self):
                pass

        sleep = AsyncMock()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key", "GEMINI_IMAGE_MODEL": "test-image-model"}), patch.object(
            genai, "Client", FakeGeminiClient
        ), patch.object(services.asyncio, "sleep", new=sleep):
            result = asyncio.run(services.analyze_image_gemini(b"image", "image/png", {}))

        self.assertEqual(result["visual_observations"], ["A visible seam appears near the object.", "Texture changes abruptly on one side of the seam."])
        self.assertEqual(FakeGeminiClient.calls, 3)
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [1.0, 2.0])

    def test_gemini_image_analysis_keeps_missing_confidence_unavailable(self):
        payload = gemini_image_payload()
        payload.pop("model_confidence")

        class FakeGeminiClient:
            def __init__(self, **kwargs): self.models = self
            def generate_content(self, **kwargs): return SimpleNamespace(text=json.dumps(payload))
            def close(self): pass

        with patch.dict(os.environ, {"GEMINI_API_KEY":"test-gemini-key"}), patch.object(genai, "Client", FakeGeminiClient):
            result = asyncio.run(services.analyze_image_gemini(b"image", "image/jpeg", {}))
        self.assertEqual(result["assessment"], "supported")
        self.assertIsNone(result["model_confidence"])

    def test_gemini_image_insufficient_evidence_forces_unavailable_scores(self):
        payload = gemini_image_payload(
            assessment="inconclusive",
            likelihood="medium",
            confidence=0.9,
            inconclusive=["Only a compressed thumbnail was available."],
        )

        class FakeGeminiClient:
            def __init__(self, **kwargs): self.models = self
            def generate_content(self, **kwargs): return SimpleNamespace(text=json.dumps(payload))
            def close(self): pass

        with patch.dict(os.environ, {"GEMINI_API_KEY":"test-gemini-key"}), patch.object(genai, "Client", FakeGeminiClient):
            result = asyncio.run(services.analyze_image_gemini(b"image", "image/jpeg", {}))
        self.assertEqual(result["manipulation_likelihood"], "unavailable")
        self.assertIsNone(result["model_confidence"])

    def test_gemini_image_duplicate_single_indicator_is_downgraded(self):
        payload = gemini_image_payload(
            assessment="suspicious",
            likelihood="high",
            indicators=["One abrupt edge is visible.", "One abrupt edge is visible."],
        )

        class FakeGeminiClient:
            def __init__(self, **kwargs): self.models = self
            def generate_content(self, **kwargs): return SimpleNamespace(text=json.dumps(payload))
            def close(self): pass

        with patch.dict(os.environ, {"GEMINI_API_KEY":"test-gemini-key"}), patch.object(genai, "Client", FakeGeminiClient):
            result = asyncio.run(services.analyze_image_gemini(b"image", "image/jpeg", {}))
        self.assertEqual(result["assessment"], "inconclusive")
        self.assertEqual(result["manipulation_likelihood"], "unavailable")
        self.assertIsNone(result["model_confidence"])
        self.assertIn("multiple distinct", result["summary"])

    def test_gemini_image_analysis_stops_after_three_retries(self):
        class ProviderError(Exception):
            code = 503

        class FakeGeminiClient:
            calls: int = 0

            def __init__(self, **kwargs):
                self.models = self

            def generate_content(self, **kwargs):
                FakeGeminiClient.calls += 1
                raise ProviderError("provider busy")

            def close(self):
                pass

        sleep = AsyncMock()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"}), patch.object(
            genai, "Client", FakeGeminiClient
        ), patch.object(services.asyncio, "sleep", new=sleep):
            with self.assertRaises(ProviderError):
                asyncio.run(services.analyze_image_gemini(b"image", "image/png", {}))

        self.assertEqual(FakeGeminiClient.calls, services.GEMINI_IMAGE_MAX_RETRIES + 1)
        self.assertEqual(sleep.await_count, services.GEMINI_IMAGE_MAX_RETRIES)

    def test_gemini_image_analysis_logs_timeout_without_provider_status(self):
        class FakeGeminiClient:
            def __init__(self, **kwargs): self.models = self
            def generate_content(self, **kwargs): raise services.httpx.ReadTimeout("timed out")
            def close(self): pass

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"}), patch.object(
            genai, "Client", FakeGeminiClient
        ), self.assertLogs(services.logger, level="ERROR") as logs:
            with self.assertRaises(services.httpx.ReadTimeout):
                asyncio.run(services.analyze_image_gemini(b"image", "image/jpeg", {}))
        joined = "\n".join(logs.output)
        self.assertIn("failure_category=network_timeout", joined)
        self.assertIn("error_type=ReadTimeout", joined)
        self.assertIn("provider_status=unavailable", joined)
        self.assertIn("request timed out before a provider response was received", joined)
        self.assertNotIn("test-gemini-key", joined)

    def test_gemini_provider_error_categories_distinguish_statuses(self):
        error = RuntimeError("provider error")
        expected = {
            401: "authentication_or_permission",
            403: "authentication_or_permission",
            404: "invalid_model_or_endpoint",
            429: "quota_or_rate_limit",
            504: "provider_deadline",
        }
        for status, category in expected.items():
            with self.subTest(status=status):
                self.assertEqual(services._gemini_failure_category(error, status), category)
        self.assertEqual(services._gemini_failure_category(services.httpx.ReadTimeout("timeout"), None), "network_timeout")

    def test_gemini_image_configuration_requires_api_key_and_sdk(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GEMINI_IMAGE_MODEL": "gemini-3.8-flash"}):
            self.assertFalse(services.gemini_image_configured())

    def test_gemini_media_analysis_uses_configured_model_and_uploaded_bytes(self):
        class FakeGeminiClient:
            request: GeminiMediaRequest | None = None
            closed: bool = False

            def __init__(self, api_key: str) -> None:
                self.api_key = api_key
                self.models = self

            def generate_content(self, *, model: str, contents: list[str | types.Part]) -> SimpleNamespace:
                FakeGeminiClient.request = {"model": model, "contents": contents}
                return SimpleNamespace(text="Observable media notes.")

            def close(self) -> None:
                FakeGeminiClient.closed = True

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key", "GEMINI_MODEL": "test-media-model"}), patch.object(
            genai, "Client", FakeGeminiClient
        ):
            result = asyncio.run(services.analyze_media_gemini(b"media bytes", "audio", "audio/wav"))
        request = required(FakeGeminiClient.request, "Gemini media request")
        self.assertEqual(result, "Observable media notes.")
        self.assertEqual(request["model"], "test-media-model")
        audio_part = request["contents"][1]
        if not isinstance(audio_part, types.Part):
            raise AssertionError("Expected a Gemini audio part")
        inline_data = required(audio_part.inline_data, "inline Gemini audio data")
        self.assertEqual(inline_data.mime_type, "audio/wav")
        self.assertTrue(FakeGeminiClient.closed)

    def test_gemini_media_analysis_requires_api_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                asyncio.run(services.analyze_media_gemini(b"media", "video", "video/mp4"))


if __name__ == "__main__":
    unittest.main()
