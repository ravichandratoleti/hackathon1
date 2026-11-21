import os
import json
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from google import genai
from google.genai import types
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# Import the application with an in-memory URL so import-time create_all cannot
# create or modify the developer's configured database.
_saved_environment = {
    name: os.environ.get(name)
    for name in ("DATABASE_URL", "VERCEL", "ENVIRONMENT")
}
os.environ["DATABASE_URL"] = "sqlite://"
os.environ.pop("VERCEL", None)
os.environ.pop("ENVIRONMENT", None)
try:
    from app import main
    from app import services
    from app.database import Base
finally:
    for _name, _value in _saved_environment.items():
        if _value is None:
            os.environ.pop(_name, None)
        else:
            os.environ[_name] = _value


class APIRoutesTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

        def override_get_db():
            db = self.session_factory()
            try:
                yield db
            finally:
                db.close()

        self.previous_overrides = dict(main.app.dependency_overrides)
        main.app.dependency_overrides[main.get_db] = override_get_db
        self.client = TestClient(main.app)

    def tearDown(self):
        self.client.close()
        main.app.dependency_overrides.clear()
        main.app.dependency_overrides.update(self.previous_overrides)
        self.engine.dispose()

    def test_health_route_and_frontend_cors_contract(self):
        response = self.client.get(
            "/api/health",
            headers={"Origin": "http://localhost:3000"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:3000")
        self.assertEqual(
            set(response.json()),
            {"status", "models", "fact_search", "fact_review"},
        )

        preflight = self.client.options(
            "/api/fact-check",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(preflight.headers["access-control-allow-origin"], "http://localhost:3000")

    def test_fact_check_without_sources_returns_unavailable_and_saves_compatible_shape(self):
        with patch.object(main, "search_sources", new=AsyncMock(return_value=[])):
            response = self.client.post(
                "/api/fact-check",
                json={"claim": "A factual claim with enough length."},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["verdict"], "insufficient_evidence")
        self.assertIn("Evidence retrieval is unavailable", body["summary"])
        self.assertEqual(body["sources"], [])
        self.assertEqual(body["evidence_coverage"], "none")
        self.assertEqual(body["claim"], "A factual claim with enough length.")
        self.assertIn("id", body)

        history = self.client.get("/api/history")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(len(history.json()), 1)
        self.assertEqual(history.json()[0]["id"], body["id"])

    def test_fact_check_returns_sources_and_verdict_in_existing_response_format(self):
        source = {
            "title": "Source title",
            "publisher": "example.org",
            "url": "https://example.org/source",
            "snippet": "Evidence snippet",
            "published_at": None,
        }
        assessment = {
            "verdict": "supported",
            "explanation": "The evidence supports the claim.",
            "findings": ["Finding from evidence."],
            "limitations": [],
        }
        assessment["model_confidence"] = 0.87
        with patch.object(main, "search_sources", new=AsyncMock(return_value=[source])), patch.object(
            main, "assess_claim", new=AsyncMock(return_value=assessment)
        ) as reviewer:
            response = self.client.post(
                "/api/fact-check",
                json={"claim": "A factual claim with enough length."},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["verdict"], "supported")
        self.assertEqual(body["sources"], [source])
        self.assertEqual(body["summary"], assessment["explanation"])
        self.assertEqual(body["evidence_coverage"], "search_snippets")
        self.assertEqual(body["model_confidence"], 0.87)
        reviewer.assert_awaited_once_with("A factual claim with enough length.", [source])

    def test_fact_check_logs_request_and_review_lifecycle_without_claim_text(self):
        claim = "Sensitive claim text that should not appear in backend logs."
        source = {"title":"Source", "publisher":"example.org", "url":"https://example.org/source", "snippet":"Evidence", "published_at":None}
        assessment = {"verdict":"supported", "explanation":"Supported by supplied evidence.", "findings":[], "limitations":[], "model_confidence":0.8}
        with patch.object(main, "search_sources", new=AsyncMock(return_value=[source])), patch.object(
            main, "assess_claim", new=AsyncMock(return_value=assessment)
        ), self.assertLogs(main.logger, level="INFO") as logs:
            response = self.client.post("/api/fact-check", json={"claim":claim})
        self.assertEqual(response.status_code, 200)
        joined_logs = "\n".join(logs.output)
        self.assertIn("Fact-check request started", joined_logs)
        self.assertIn("source_count=1", joined_logs)
        self.assertIn("Fact-check request completed", joined_logs)
        self.assertNotIn(claim, joined_logs)

    def test_fact_check_tavily_failure_does_not_call_verdict_reviewer(self):
        from app.services import EvidenceRetrievalError
        with patch.object(main, "search_sources", new=AsyncMock(side_effect=EvidenceRetrievalError("Evidence retrieval is unavailable."))), patch.object(
            main, "assess_claim", new=AsyncMock()) as reviewer:
            response = self.client.post("/api/fact-check", json={"claim": "A factual claim with enough length."})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["verdict"], "insufficient_evidence")
        self.assertEqual(response.json()["sources"], [])
        reviewer.assert_not_awaited()

    def test_fact_check_empty_results_never_generates_verdict(self):
        with patch.object(main, "search_sources", new=AsyncMock(return_value=[])), patch.object(
            main, "assess_claim", new=AsyncMock(return_value={"verdict":"insufficient_evidence", "explanation":"Evidence retrieval is unavailable.", "findings":[], "limitations":[]})):
            response = self.client.post("/api/fact-check", json={"claim": "A factual claim with enough length."})
        self.assertEqual(response.json()["verdict"], "insufficient_evidence")
        self.assertEqual(response.json()["verdict"], "insufficient_evidence")

    def test_gemini_review_failure_is_never_returned_as_a_successful_verdict(self):
        source = {"title": "Source", "publisher": "example.org", "url": "https://example.org", "snippet": "Evidence", "provider": "Tavily"}
        with patch.object(main, "search_sources", new=AsyncMock(return_value=[source])), patch.object(
            main, "assess_claim", new=AsyncMock(return_value={"verdict":"review_unavailable", "explanation":"Evidence was retrieved, but verdict review is unavailable.", "findings":[], "limitations":["Gemini verdict review is unavailable (authentication_or_permission); no factual verdict was generated."], "model_confidence":None})):
            response = self.client.post("/api/fact-check", json={"claim": "A factual claim with enough length."})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["verdict"], "review_unavailable")
        self.assertEqual(body["evidence_coverage"], "search_snippets")
        self.assertEqual(body["sources"], [source])
        self.assertIn("review is unavailable", body["summary"])
        self.assertTrue(any("no factual verdict" in item.lower() for item in body["limitations"]))

    def test_invalid_claim_is_rejected_before_provider_calls(self):
        with patch.object(main, "search_sources", new=AsyncMock()) as search:
            response = self.client.post("/api/fact-check", json={"claim": "short"})

        self.assertEqual(response.status_code, 400)
        search.assert_not_awaited()

    def test_image_route_returns_structured_gemini_observations(self):
        image = BytesIO()
        Image.new("RGB", (2, 2), (255, 255, 255)).save(image, format="PNG")
        gemini_result = {
            "visual_observations": ["Possible repeated texture; inspect original at full resolution."],
            "metadata_observations": ["No selected EXIF fields were supplied."],
            "inconclusive_reasons": ["A single image cannot establish editing history."],
        }
        with patch.object(main, "gemini_image_configured", return_value=True), patch.object(
            main, "analyze_image_gemini", new=AsyncMock(return_value=gemini_result)
        ) as analyze:
            response = self.client.post(
                "/api/analyze/image",
                files={"file": ("sample.png", image.getvalue(), "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["analysis_type"], "image")
        self.assertEqual(body["verdict"], "inconclusive")
        self.assertIsNone(body["manipulation_likelihood"])
        self.assertIsNone(body["model_confidence"])
        self.assertEqual(body["evidence_coverage"], "gemini_image_review")
        self.assertIn("Possible repeated texture", body["findings"][2]["text"])
        self.assertEqual(body["findings"][2]["type"], "visual_observation")
        self.assertEqual(body["findings"][3]["type"], "metadata_observation")
        self.assertEqual(body["findings"][4]["type"], "inconclusive")
        analyze.assert_awaited_once()
        self.assertEqual(analyze.await_args.args[1], "image/png")

    def test_image_endpoint_upload_runs_gemini_image_integration(self):
        class FakeGeminiClient:
            request: dict[str, object] | None = None
            http_options: types.HttpOptions | None = None

            def __init__(self, *, api_key: str, http_options: types.HttpOptions) -> None:
                FakeGeminiClient.http_options = http_options
                self.models = self

            def generate_content(self, **kwargs: object) -> SimpleNamespace:
                FakeGeminiClient.request = kwargs
                return SimpleNamespace(text=json.dumps({
                    "visual_observations": ["A possible edge discontinuity is visible."],
                    "metadata_observations": ["No selected EXIF fields were available."],
                    "inconclusive_reasons": ["A single image cannot establish editing history."],
                }))

            def close(self) -> None:
                pass

        image = BytesIO()
        Image.new("RGB", (8, 6), (120, 140, 160)).save(image, format="PNG")
        uploaded_image = image.getvalue()
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "test-gemini-key",
            "GEMINI_IMAGE_MODEL": "test-image-forensics-model",
        }), patch.object(main, "gemini_image_configured", return_value=True), patch.object(
            genai, "Client", FakeGeminiClient
        ):
            response = self.client.post(
                "/api/analyze/image",
                files={"file": ("uploaded.png", uploaded_image, "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["analysis_type"], "image")
        self.assertEqual(body["verdict"], "inconclusive")
        self.assertIsNone(body["model_confidence"])
        self.assertEqual(body["evidence_coverage"], "gemini_image_review")
        self.assertIn("possible edge discontinuity", body["findings"][2]["text"])
        request = FakeGeminiClient.request
        self.assertIsNotNone(request)
        assert request is not None
        model = request.get("model")
        self.assertEqual(model, "test-image-forensics-model")
        request_contents = request["contents"]
        self.assertIsInstance(request_contents, list)
        if not isinstance(request_contents, list) or len(request_contents) < 2:
            raise AssertionError("Expected prompt and uploaded image parts")
        image_part = request_contents[1]
        self.assertIsInstance(image_part, types.Part)
        if not isinstance(image_part, types.Part) or image_part.inline_data is None:
            raise AssertionError("Expected inline image data in the Gemini request")
        self.assertEqual(image_part.inline_data.data, uploaded_image)
        self.assertEqual(image_part.inline_data.mime_type, "image/png")
        request_config = request.get("config")
        self.assertIsInstance(request_config, types.GenerateContentConfig)
        if not isinstance(request_config, types.GenerateContentConfig):
            raise AssertionError("Expected the production structured JSON configuration")
        self.assertEqual(request_config.response_mime_type, "application/json")
        self.assertIs(request_config.response_schema, services.GeminiImageForensics)
        self.assertIsNotNone(FakeGeminiClient.http_options)
        if FakeGeminiClient.http_options is None:
            raise AssertionError("Expected Gemini HTTP client options")
        self.assertEqual(FakeGeminiClient.http_options.timeout, 90_000)

    def test_image_route_rejects_unreadable_file(self):
        response = self.client.post(
            "/api/analyze/image",
            files={"file": ("sample.png", b"not an image", "image/png")},
        )

        self.assertEqual(response.status_code, 400)

    def test_image_provider_failure_keeps_result_inconclusive_without_scores(self):
        image = BytesIO()
        Image.new("RGB", (2, 2), (255, 255, 255)).save(image, format="PNG")
        with patch.object(main, "gemini_image_configured", return_value=True), patch.object(
            main, "analyze_image_gemini", new=AsyncMock(side_effect=RuntimeError("provider unavailable"))
        ):
            response = self.client.post(
                "/api/analyze/image",
                files={"file": ("sample.png", image.getvalue(), "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["verdict"], "inconclusive")
        self.assertIsNone(body["model_confidence"])
        self.assertEqual(body["evidence_coverage"], "metadata_only")
        self.assertNotIn("visual_observation", [finding["type"] for finding in body["findings"]])

    def test_image_health_uses_gemini_configuration(self):
        with patch.object(main, "gemini_image_configured", return_value=True):
            response = self.client.get("/api/health")
        self.assertTrue(response.json()["models"]["image"])

    def test_video_and_audio_routes_call_gemini_and_remain_inconclusive(self):
        with patch.object(main, "gemini_media_configured", return_value=True), patch.object(
            main, "analyze_media_gemini", new=AsyncMock(return_value="Observed content details.")
        ) as analyze:
            video = self.client.post(
                "/api/analyze/video",
                files={"file": ("sample.mp4", b"video placeholder", "video/mp4")},
            )
            audio = self.client.post(
                "/api/analyze/audio",
                files={"file": ("sample.wav", b"audio placeholder", "audio/wav")},
            )

        self.assertEqual(video.status_code, 200)
        self.assertEqual(audio.status_code, 200)
        self.assertEqual(video.json()["verdict"], "inconclusive")
        self.assertEqual(audio.json()["verdict"], "inconclusive")
        self.assertIn("Observed content details.", video.json()["findings"][0]["text"])
        self.assertIn("Observed content details.", audio.json()["findings"][0]["text"])
        self.assertEqual(analyze.await_count, 2)

    def test_media_routes_report_provider_failure_without_model_result(self):
        with patch.object(main, "gemini_media_configured", return_value=True), patch.object(
            main, "analyze_media_gemini", new=AsyncMock(side_effect=RuntimeError("provider unavailable"))
        ):
            response = self.client.post(
                "/api/analyze/audio",
                files={"file": ("sample.wav", b"audio placeholder", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["verdict"], "inconclusive")
        self.assertEqual(response.json()["evidence_coverage"], "analysis_failed")
        self.assertIn("failed", response.json()["summary"].lower())
        self.assertEqual(response.json()["findings"], [])

    def test_health_reports_gemini_media_configuration(self):
        with patch.object(main, "gemini_media_configured", return_value=True):
            response = self.client.get("/api/health")
        self.assertTrue(response.json()["models"]["video"])
        self.assertTrue(response.json()["models"]["audio"])


if __name__ == "__main__":
    unittest.main()
