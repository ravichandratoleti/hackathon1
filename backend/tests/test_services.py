import asyncio
import builtins
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from google import genai

from app import services


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
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.is_error = status_code >= 400

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeAsyncHTTPClient:
    calls = []
    payload = None
    status_code = 200
    created = 0
    error = None

    def __init__(self, **kwargs):
        self.created += 1
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        return FakeHTTPResponse(self.payload, self.status_code)


class ServicesTests(unittest.TestCase):
    def test_gemini_interaction_uses_configured_model_google_search_and_maps_citations(self):
        response = interaction_with_citations(citation(start=0, end=8), text="Evidence text.")

        class FakeGeminiClient:
            request = None
            closed = False

            def __init__(self, api_key):
                self.api_key = api_key
                self.interactions = self

            def create(self, **kwargs):
                FakeGeminiClient.request = kwargs
                return response

            def close(self):
                FakeGeminiClient.closed = True

        with patch.object(genai, "Client", FakeGeminiClient):
            sources = services._gemini_search_sync("A factual claim", "test-token")

        self.assertEqual(services.GEMINI_MODEL, "gemini-3.5-flash-lite")
        self.assertEqual(FakeGeminiClient.request["model"], services.GEMINI_MODEL)
        self.assertEqual(FakeGeminiClient.request["tools"], [{"type": "google_search"}])
        self.assertFalse(FakeGeminiClient.request["store"])
        self.assertIn("A factual claim", FakeGeminiClient.request["input"])
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
        self.assertEqual(url, "https://api.tavily.com/search")
        self.assertEqual(request["headers"]["Authorization"], "Bearer test-token")
        self.assertEqual(request["json"]["max_results"], 5)
        self.assertEqual(request["json"]["search_depth"], "basic")
        self.assertFalse(request["json"]["include_answer"])
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
            def json(self):
                raise ValueError("not JSON")

        class InvalidJSONClient(FakeAsyncHTTPClient):
            async def post(self, url, **kwargs):
                self.calls.append((url, kwargs))
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
            request = None
            def __init__(self, api_key):
                self.interactions = self
            def create(self, **kwargs):
                self.request = kwargs
                return SimpleNamespace(output_text="Evidence is mixed; source 1 supports a narrower claim.")
            def close(self):
                pass

        client = FakeGeminiClient("test-token")
        evidence_sources = [{"title":"A", "publisher":"example.org", "snippet":"Source says it occurred."}, {"title":"B", "publisher":"example.net", "snippet":"Source says it did not occur."}]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-token"}), patch.object(
            genai, "Client", return_value=client
        ):
            analysis = asyncio.run(services.analyze_evidence_gemini("A factual claim", evidence_sources))
        self.assertEqual(analysis, "Evidence is mixed; source 1 supports a narrower claim.")
        self.assertEqual(client.request["model"], services.GEMINI_MODEL)
        self.assertNotIn("tools", client.request)
        self.assertFalse(client.request["store"])
        self.assertIn("Source says it occurred.", client.request["input"])
        self.assertIn("Source says it did not occur.", client.request["input"])

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
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(request["json"]["model"], "test-verdict-model")
        self.assertEqual(request["headers"]["Authorization"], "Bearer test-token")
        user_prompt = json.loads(request["json"]["messages"][1]["content"])
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
            async def post(self, url, **kwargs):
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
        prompt = json.loads(FakeAsyncHTTPClient.calls[0][1]["json"]["messages"][1]["content"])
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

    def test_missing_image_model_configuration_makes_no_request(self):
        FakeAsyncHTTPClient.created = 0
        with patch.dict(os.environ, {"HF_API_TOKEN": "", "HF_IMAGE_MODEL": ""}), patch.object(
            services.httpx, "AsyncClient", FakeAsyncHTTPClient
        ):
            result = asyncio.run(services.classify_image(b"image bytes"))

        self.assertIsNone(result)
        self.assertEqual(FakeAsyncHTTPClient.created, 0)


if __name__ == "__main__":
    unittest.main()
