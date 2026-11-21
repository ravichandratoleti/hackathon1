import os
import unittest
from io import BytesIO
from unittest.mock import AsyncMock, patch

import httpx
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
        with patch.object(main, "search_sources", new=AsyncMock(return_value=[source])), patch.object(
            main, "analyze_evidence_gemini", new=AsyncMock(return_value="Preliminary")), patch.object(
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
        reviewer.assert_awaited_once_with("A factual claim with enough length.", [source], gemini_analysis="Preliminary")

    def test_fact_check_tavily_failure_is_unavailable_and_does_not_call_gemini_or_openai(self):
        from app.services import EvidenceRetrievalError
        with patch.object(main, "search_sources", new=AsyncMock(side_effect=EvidenceRetrievalError("Evidence retrieval is unavailable."))), patch.object(
            main, "analyze_evidence_gemini", new=AsyncMock()) as gemini, patch.object(
            main, "assess_claim", new=AsyncMock()) as openai:
            response = self.client.post("/api/fact-check", json={"claim": "A factual claim with enough length."})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["verdict"], "insufficient_evidence")
        self.assertEqual(response.json()["sources"], [])
        gemini.assert_not_awaited()
        openai.assert_not_awaited()

    def test_fact_check_empty_results_never_generates_verdict(self):
        with patch.object(main, "search_sources", new=AsyncMock(return_value=[])), patch.object(
            main, "analyze_evidence_gemini", new=AsyncMock()) as gemini, patch.object(
            main, "assess_claim", new=AsyncMock(return_value={"verdict":"insufficient_evidence", "explanation":"Evidence retrieval is unavailable.", "findings":[], "limitations":[]})):
            response = self.client.post("/api/fact-check", json={"claim": "A factual claim with enough length."})
        self.assertEqual(response.json()["verdict"], "insufficient_evidence")
        gemini.assert_not_awaited()

    def test_gemini_failure_is_reported_as_limitation_while_openai_reviews_real_sources(self):
        source = {"title":"Source", "publisher":"example.org", "url":"https://example.org", "snippet":"Actual snippet", "provider":"Tavily"}
        assessment = {"verdict":"insufficient_evidence", "explanation":"The evidence is insufficient.", "findings":[], "limitations":[]}
        with patch.object(main, "search_sources", new=AsyncMock(return_value=[source])), patch.object(
            main, "analyze_evidence_gemini", new=AsyncMock(side_effect=RuntimeError("mock failure"))
        ), patch.object(main, "assess_claim", new=AsyncMock(return_value=assessment)) as reviewer:
            response = self.client.post("/api/fact-check", json={"claim":"A factual claim with enough length."})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"], [source])
        reviewer.assert_awaited_once_with("A factual claim with enough length.", [source], gemini_analysis=None)
        self.assertTrue(any("Gemini preliminary evidence analysis was unavailable" in x for x in response.json()["limitations"]))

    def test_openai_review_failure_is_never_returned_as_a_successful_verdict(self):
        source = {"title": "Source", "publisher": "example.org", "url": "https://example.org", "snippet": "Evidence", "provider": "Tavily"}
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        provider_response = httpx.Response(401, json={"error": {"code": "invalid_api_key"}}, request=request)
        failure = httpx.HTTPStatusError("HTTP 401", request=request, response=provider_response)
        with patch.object(main, "search_sources", new=AsyncMock(return_value=[source])), patch.object(
            main, "analyze_evidence_gemini", new=AsyncMock(return_value="Preliminary")
        ), patch.object(main, "assess_claim", new=AsyncMock(side_effect=failure)):
            response = self.client.post("/api/fact-check", json={"claim": "A factual claim with enough length."})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["verdict"], "insufficient_evidence")
        self.assertEqual(body["evidence_coverage"], "search_snippets")
        self.assertEqual(body["sources"], [source])
        self.assertIn("review is unavailable", body["summary"])
        self.assertTrue(any("no factual verdict" in item.lower() for item in body["limitations"]))

    def test_invalid_claim_is_rejected_before_provider_calls(self):
        with patch.object(main, "search_sources", new=AsyncMock()) as search:
            response = self.client.post("/api/fact-check", json={"claim": "short"})

        self.assertEqual(response.status_code, 400)
        search.assert_not_awaited()

    def test_image_route_validates_and_returns_classifier_result(self):
        image = BytesIO()
        Image.new("RGB", (2, 2), (255, 255, 255)).save(image, format="PNG")
        classifier_result = {
            "model": "mock-image-model",
            "scores": [{"label": "AI-generated", "score": 0.91}],
        }
        with patch.dict(os.environ, {"HF_API_TOKEN": "", "HF_IMAGE_MODEL": ""}), patch.object(
            main, "classify_image", new=AsyncMock(return_value=classifier_result)
        ):
            response = self.client.post(
                "/api/analyze/image",
                files={"file": ("sample.png", image.getvalue(), "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["analysis_type"], "image")
        self.assertEqual(body["verdict"], "potentially_manipulated")
        self.assertIn("mock-image-model", body["findings"][1]["text"])

    def test_image_route_rejects_unreadable_file(self):
        response = self.client.post(
            "/api/analyze/image",
            files={"file": ("sample.png", b"not an image", "image/png")},
        )

        self.assertEqual(response.status_code, 400)

    def test_video_and_audio_routes_preserve_unavailable_state(self):
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
        self.assertIn("unavailable", video.json()["limitations"][0].lower())
        self.assertIn("unavailable", audio.json()["limitations"][0].lower())


if __name__ == "__main__":
    unittest.main()
