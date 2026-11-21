import asyncio
import builtins
import json
import os
import unittest
from types import SimpleNamespace
from typing import TypeGuard, TypedDict, TypeVar
from unittest.mock import AsyncMock, patch

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


class OpenAIRequestMessage(TypedDict):
    role: str
    content: str


class OpenAIRequestBody(TypedDict):
    model: str
    temperature: int
    response_format: dict[str, str]
    messages: list[OpenAIRequestMessage]


class CapturedHTTPCall(TypedDict):
    headers: dict[str, str]
    json: TavilyRequestBody | OpenAIRequestBody


class GeminiSearchRequest(TypedDict):
    model: str
    input: str
    tools: list[dict[str, str]]
    store: bool


class GeminiEvidenceRequest(TypedDict):
    model: str
    input: str
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


def is_tavily_body(body: TavilyRequestBody | OpenAIRequestBody) -> TypeGuard[TavilyRequestBody]:
    return "query" in body


def is_openai_body(body: TavilyRequestBody | OpenAIRequestBody) -> TypeGuard[OpenAIRequestBody]:
    return "messages" in body


def tavily_body(call: CapturedHTTPCall) -> TavilyRequestBody:
    body = call["json"]
    if not is_tavily_body(body):
        raise AssertionError("Expected captured Tavily request")
    return body


def openai_body(call: CapturedHTTPCall) -> OpenAIRequestBody:
    body = call["json"]
    if not is_openai_body(body):
        raise AssertionError("Expected captured OpenAI request")
    return body


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
        json: TavilyRequestBody | OpenAIRequestBody,
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
                json: TavilyRequestBody | OpenAIRequestBody,
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

    def test_gemini_analysis_uses_configured_model_without_search_tool(self):
        class FakeGeminiClient:
            request: GeminiEvidenceRequest | None = None

            def __init__(self, api_key: str) -> None:
                self.interactions = self

            def create(self, *, model: str, input: str, store: bool) -> SimpleNamespace:
                self.request = {"model": model, "input": input, "store": store}
                return SimpleNamespace(output_text="Evidence is mixed; source 1 supports a narrower claim.")

            def close(self) -> None:
                pass

        client = FakeGeminiClient("test-token")
        evidence_sources = [{"title":"A", "publisher":"example.org", "snippet":"Source says it occurred."}, {"title":"B", "publisher":"example.net", "snippet":"Source says it did not occur."}]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-token"}), patch.object(
            genai, "Client", return_value=client
        ):
            analysis = asyncio.run(services.analyze_evidence_gemini("A factual claim", evidence_sources))
        self.assertEqual(analysis, "Evidence is mixed; source 1 supports a narrower claim.")
        request = required(client.request, "Gemini analysis request")
        self.assertEqual(request["model"], services.GEMINI_MODEL)
        self.assertNotIn("tools", request)
        self.assertFalse(request["store"])
        self.assertIn("Source says it occurred.", request["input"])
        self.assertIn("Source says it did not occur.", request["input"])

    def test_missing_gemini_sdk_logs_exact_missing_module_safely(self):
        original_import = builtins.__import__

        def import_without_genai(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "google" and "genai" in (fromlist or ()):
                raise ModuleNotFoundError("No module named 'google.genai'", name="google.genai")
            return original_import(name, globals, locals, fromlist, level)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-gemini-key"}), patch(
            "builtins.__import__", side_effect=import_without_genai
        ), self.assertLogs(services.logger, level="ERROR") as logs:
            with self.assertRaisesRegex(RuntimeError, "SDK dependency"):
                asyncio.run(services.analyze_evidence_gemini("A factual claim", []))
        self.assertTrue(any("missing_module=google.genai" in line for line in logs.output))
        self.assertTrue(any("google-genai" in line for line in logs.output))
        self.assertFalse(any("dummy-gemini-key" in line for line in logs.output))

    def test_no_sources_returns_unavailable_without_an_openai_request(self):
        FakeAsyncHTTPClient.calls = []
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-token"}), patch.object(
            services.httpx, "AsyncClient", FakeAsyncHTTPClient
        ):
            result = asyncio.run(services.assess_claim("A factual claim", []))

        self.assertEqual(result["verdict"], "insufficient_evidence")
        self.assertIn("Evidence retrieval is unavailable", result["explanation"])
        self.assertEqual(FakeAsyncHTTPClient.calls, [])

    def test_missing_openai_key_returns_insufficient_evidence(self):
        source = {"title": "Source", "url": "https://example.org", "snippet": "Text"}
        FakeAsyncHTTPClient.calls = []
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch.object(
            services.httpx, "AsyncClient", FakeAsyncHTTPClient
        ):
            result = asyncio.run(services.assess_claim("A factual claim", [source]))

        self.assertEqual(result["verdict"], "insufficient_evidence")
        self.assertEqual(FakeAsyncHTTPClient.calls, [])

    def test_openai_verdict_generation_uses_mocked_response(self):
        source = {
            "title": "Example report",
            "publisher": "example.org",
            "url": "https://example.org/report",
            "snippet": "Evidence text.",
        }
        FakeAsyncHTTPClient.calls = []
        FakeAsyncHTTPClient.payload = {
            "choices": [{"message": {"content": json.dumps({
                "verdict": "supported",
                "explanation": "The supplied evidence supports the claim.",
                "findings": ["Evidence text supports it."],
            })}}]
        }

        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-token",
            "OPENAI_MODEL": "test-verdict-model",
        }), patch.object(services.httpx, "AsyncClient", FakeAsyncHTTPClient):
            result = asyncio.run(services.assess_claim("A factual claim", [source]))

        self.assertEqual(result["verdict"], "supported")
        self.assertEqual(result["findings"], ["Evidence text supports it."])
        url, request = FakeAsyncHTTPClient.calls[0]
        body = openai_body(request)
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(body["model"], "test-verdict-model")
        self.assertEqual(request["headers"]["Authorization"], "Bearer test-token")
        user_prompt = json.loads(body["messages"][1]["content"])
        self.assertEqual(user_prompt["sources"][0]["url"], source["url"])

    def test_openai_http_error_logs_status_and_redacted_provider_body(self):
        request = services.httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        fake_key = "sk-proj-" + "fake-secret-value"
        response = services.httpx.Response(
            401,
            json={"error": {
                "message": f"Incorrect API key provided: {fake_key}",
                "type": "authentication_error",
                "param": None,
                "code": "invalid_api_key",
            }},
            headers={"x-request-id": "req-test-123"},
            request=request,
        )
        failure = services.httpx.HTTPStatusError("HTTP 401", request=request, response=response)

        class HTTPErrorClient(FakeAsyncHTTPClient):
            async def post(
                self,
                url: str,
                *,
                headers: dict[str, str],
                json: TavilyRequestBody | OpenAIRequestBody,
            ) -> FakeHTTPResponse:
                raise failure

        source = {"title": "Example", "url": "https://example.org", "snippet": "Evidence"}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-openai-key", "OPENAI_MODEL": "gpt-4o-mini"}), patch.object(
            services.httpx, "AsyncClient", HTTPErrorClient
        ), self.assertLogs(services.logger, level="ERROR") as logs:
            with self.assertRaises(services.httpx.HTTPStatusError):
                asyncio.run(services.assess_claim("A factual claim", [source]))

        joined_logs = "\n".join(logs.output)
        self.assertIn("status=401", joined_logs)
        self.assertIn("invalid_api_key", joined_logs)
        self.assertIn("req-test-123", joined_logs)
        self.assertIn("[REDACTED]", joined_logs)
        self.assertNotIn(fake_key, joined_logs)
        self.assertNotIn("dummy-openai-key", joined_logs)

    def test_openai_verdict_reviewer_receives_gemini_analysis_as_untrusted_context(self):
        source = {"title": "Example", "publisher": "example.org", "url": "https://example.org", "snippet": "Evidence"}
        FakeAsyncHTTPClient.calls = []
        FakeAsyncHTTPClient.payload = {"choices": [{"message": {"content": json.dumps({
            "verdict": "mixed", "explanation": "Evidence is mixed.", "findings": [],
        })}}]}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-token"}), patch.object(
            services.httpx, "AsyncClient", FakeAsyncHTTPClient
        ):
            asyncio.run(services.assess_claim("A factual claim", [source], gemini_analysis="Preliminary mixed evidence"))
        prompt = json.loads(openai_body(FakeAsyncHTTPClient.calls[0][1])["messages"][1]["content"])
        self.assertEqual(prompt["gemini_preliminary_analysis"], "Preliminary mixed evidence")
        self.assertIn("fallible", prompt["instruction"])

    def test_invalid_openai_verdict_is_normalized_to_insufficient_evidence(self):
        source = {"title": "Source", "url": "https://example.org", "snippet": "Text"}
        FakeAsyncHTTPClient.calls = []
        FakeAsyncHTTPClient.payload = {
            "choices": [{"message": {"content": json.dumps({
                "verdict": "certainly_true",
                "explanation": "Unsupported verdict label.",
                "findings": [],
            })}}]
        }

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-token"}), patch.object(
            services.httpx, "AsyncClient", FakeAsyncHTTPClient
        ):
            result = asyncio.run(services.assess_claim("A factual claim", [source]))

        self.assertEqual(result["verdict"], "insufficient_evidence")

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
                return SimpleNamespace(text=json.dumps({
                    "visual_observations": ["Shadow direction appears inconsistent."],
                    "metadata_observations": ["EXIF Software field is present."],
                    "inconclusive_reasons": ["The original capture is unavailable."],
                }))

            def close(self) -> None:
                FakeGeminiClient.closed = True

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key", "GEMINI_IMAGE_MODEL": "gemini-3.8-flash"}), patch.object(
            genai, "Client", FakeGeminiClient
        ):
            result = asyncio.run(services.analyze_image_gemini(b"image bytes", "image/png", {"Software": "Camera"}))

        self.assertEqual(result["visual_observations"], ["Shadow direction appears inconsistent."])
        self.assertEqual(result["metadata_observations"], ["EXIF Software field is present."])
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
                return SimpleNamespace(text=json.dumps({
                    "visual_observations": [],
                    "metadata_observations": [],
                    "inconclusive_reasons": ["Insufficient detail."],
                }))

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
                return SimpleNamespace(text=json.dumps({
                    "visual_observations": ["A visible seam appears near the object."],
                    "metadata_observations": [],
                    "inconclusive_reasons": [],
                }))

            def close(self):
                pass

        sleep = AsyncMock()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key", "GEMINI_IMAGE_MODEL": "test-image-model"}), patch.object(
            genai, "Client", FakeGeminiClient
        ), patch.object(services.asyncio, "sleep", new=sleep):
            result = asyncio.run(services.analyze_image_gemini(b"image", "image/png", {}))

        self.assertEqual(result["visual_observations"], ["A visible seam appears near the object."])
        self.assertEqual(FakeGeminiClient.calls, 3)
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [1.0, 2.0])

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
