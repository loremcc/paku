from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from paku.web.app import create_app


def _png_bytes(color: tuple[int, int, int] = (120, 120, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _fake_anime_result(filename: str, *, anilist_id: int = 1234, title: str = "Frieren") -> dict:
    return {
        "screenshot": filename,
        "screen_type": "post",
        "content_type": "anime",
        "ocr_text": f"{title} — episode 1",
        "engine": "stub",
        "outputs": [],
        "smart": False,
        "extracted_at": "2026-04-22T00:00:00+00:00",
        "status": "extracted",
        "extraction": {
            "extractor": "anime",
            "confidence": 0.92,
            "needs_review": False,
            "source_screenshot": filename,
            "raw_title": title,
            "canonical_title": title,
            "romaji": title,
            "native_title": None,
            "media_format": "TV",
            "source": "MANGA",
            "country_of_origin": "Japan",
            "debut_year": 2023,
            "studios": ["Madhouse"],
            "genres": ["Adventure", "Fantasy"],
            "score": 9.2,
            "episodes": 28,
            "status": "FINISHED",
            "cover_image": "https://example.com/cover.jpg",
            "banner_image": None,
            "anilist_url": f"https://anilist.co/anime/{anilist_id}",
            "anilist_id": anilist_id,
            "extraction_mode": "fast",
        },
    }


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app(db_path=tmp_path / "web.db")
    return TestClient(app)


@pytest.fixture
def anime_png() -> bytes:
    return _png_bytes()


# --- /api/digest ---


class TestDigestEndpoint:
    def test_digest_anime_success(self, client: TestClient, anime_png: bytes) -> None:
        with patch("paku.web.app.process_image") as mock_proc:
            mock_proc.return_value = _fake_anime_result("shot.png", anilist_id=111)
            resp = client.post(
                "/api/digest",
                files={"file": ("shot.png", anime_png, "image/png")},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["content_type"] == "anime"
        assert len(body["stored"]) == 1
        assert body["stored"][0]["anilist_id"] == 111
        assert body["needs_review"] is False

    def test_digest_empty_file(self, client: TestClient) -> None:
        resp = client.post(
            "/api/digest",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 422

    def test_digest_unprocessable(self, client: TestClient, anime_png: bytes) -> None:
        with patch("paku.web.app.process_image", return_value=None):
            resp = client.post(
                "/api/digest",
                files={"file": ("shot.png", anime_png, "image/png")},
            )
        assert resp.status_code == 422

    def test_digest_pipeline_error(self, client: TestClient, anime_png: bytes) -> None:
        with patch("paku.web.app.process_image", side_effect=RuntimeError("boom")):
            resp = client.post(
                "/api/digest",
                files={"file": ("shot.png", anime_png, "image/png")},
            )
        assert resp.status_code == 422
        assert "boom" in resp.json()["detail"]

    def test_digest_then_collection(self, client: TestClient, anime_png: bytes) -> None:
        with patch("paku.web.app.process_image") as mock_proc:
            mock_proc.return_value = _fake_anime_result("one.png", anilist_id=200)
            client.post(
                "/api/digest", files={"file": ("one.png", anime_png, "image/png")}
            )
        resp = client.get("/api/collection")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["anilist_id"] == 200

    def test_digest_idempotent(self, client: TestClient, anime_png: bytes) -> None:
        result = _fake_anime_result("same.png", anilist_id=300)
        with patch("paku.web.app.process_image", return_value=result):
            client.post(
                "/api/digest", files={"file": ("same.png", anime_png, "image/png")}
            )
            client.post(
                "/api/digest", files={"file": ("same.png", anime_png, "image/png")}
            )
        assert client.get("/api/collection").json()["total"] == 1


# --- /api/collection ---


class TestCollectionEndpoints:
    def test_empty_collection(self, client: TestClient) -> None:
        resp = client.get("/api/collection")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["items"] == []

    def test_get_missing_item(self, client: TestClient) -> None:
        resp = client.get("/api/collection/999")
        assert resp.status_code == 404

    def test_patch_updates_user_status(
        self, client: TestClient, anime_png: bytes
    ) -> None:
        with patch("paku.web.app.process_image") as mock_proc:
            mock_proc.return_value = _fake_anime_result("s.png", anilist_id=1)
            digest = client.post(
                "/api/digest", files={"file": ("s.png", anime_png, "image/png")}
            ).json()
        anime_id = digest["stored"][0]["id"]

        resp = client.patch(
            f"/api/collection/{anime_id}", json={"user_status": "Watching"}
        )
        assert resp.status_code == 200
        assert resp.json()["user_status"] == "Watching"

    def test_patch_invalid_status(
        self, client: TestClient, anime_png: bytes
    ) -> None:
        with patch("paku.web.app.process_image") as mock_proc:
            mock_proc.return_value = _fake_anime_result("s.png", anilist_id=1)
            digest = client.post(
                "/api/digest", files={"file": ("s.png", anime_png, "image/png")}
            ).json()
        anime_id = digest["stored"][0]["id"]

        resp = client.patch(
            f"/api/collection/{anime_id}", json={"user_status": "Bogus"}
        )
        assert resp.status_code == 422

    def test_patch_missing_item(self, client: TestClient) -> None:
        resp = client.patch(
            "/api/collection/999", json={"user_status": "Watching"}
        )
        assert resp.status_code == 404

    def test_delete(self, client: TestClient, anime_png: bytes) -> None:
        with patch("paku.web.app.process_image") as mock_proc:
            mock_proc.return_value = _fake_anime_result("s.png", anilist_id=1)
            digest = client.post(
                "/api/digest", files={"file": ("s.png", anime_png, "image/png")}
            ).json()
        anime_id = digest["stored"][0]["id"]

        resp = client.delete(f"/api/collection/{anime_id}")
        assert resp.status_code == 200
        assert client.get("/api/collection").json()["total"] == 0

    def test_delete_missing(self, client: TestClient) -> None:
        assert client.delete("/api/collection/999").status_code == 404

    def test_clear_review(self, client: TestClient, anime_png: bytes) -> None:
        result = _fake_anime_result("s.png", anilist_id=1)
        result["extraction"]["needs_review"] = True
        with patch("paku.web.app.process_image", return_value=result):
            digest = client.post(
                "/api/digest", files={"file": ("s.png", anime_png, "image/png")}
            ).json()
        anime_id = digest["stored"][0]["id"]

        resp = client.post(f"/api/collection/{anime_id}/review/clear")
        assert resp.status_code == 200
        assert resp.json()["needs_review"] is False


# --- /api/search ---


class TestSearchEndpoints:
    def test_search_anilist(self, client: TestClient) -> None:
        fake_page = {
            "data": {
                "Page": {
                    "media": [
                        {
                            "id": 51,
                            "title": {
                                "english": "Frieren",
                                "romaji": "Sousou no Frieren",
                                "native": "葬送のフリーレン",
                            },
                            "type": "ANIME",
                            "format": "TV",
                            "episodes": 28,
                            "status": "FINISHED",
                            "genres": ["Adventure"],
                            "averageScore": 92,
                            "siteUrl": "https://anilist.co/anime/51",
                            "countryOfOrigin": "JP",
                            "startDate": {"year": 2023},
                            "coverImage": {
                                "extraLarge": "https://e.com/x.jpg",
                                "large": "https://e.com/l.jpg",
                            },
                            "bannerImage": None,
                            "source": "MANGA",
                            "studios": {
                                "edges": [
                                    {
                                        "node": {
                                            "name": "Madhouse",
                                            "isAnimationStudio": True,
                                        }
                                    }
                                ]
                            },
                        }
                    ]
                }
            }
        }

        class FakeResp:
            status_code = 200

            def raise_for_status(self) -> None: ...

            def json(self) -> dict:
                return fake_page

        with patch("paku.web.app.requests.post", return_value=FakeResp()):
            resp = client.get("/api/search", params={"q": "Frieren"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "Frieren"
        assert len(body["results"]) == 1
        assert body["results"][0]["anilist_id"] == 51
        assert body["results"][0]["score"] == pytest.approx(9.2)

    def test_search_add(self, client: TestClient) -> None:
        fake_media = {
            "data": {
                "Media": {
                    "id": 77,
                    "title": {
                        "english": "Test",
                        "romaji": "Test",
                        "native": None,
                    },
                    "type": "ANIME",
                    "format": "TV",
                    "episodes": 12,
                    "status": "FINISHED",
                    "genres": ["Drama"],
                    "averageScore": 80,
                    "siteUrl": "https://anilist.co/anime/77",
                    "countryOfOrigin": "JP",
                    "startDate": {"year": 2020},
                    "coverImage": {"extraLarge": None, "large": None},
                    "bannerImage": None,
                    "source": "ORIGINAL",
                    "studios": {"edges": []},
                }
            }
        }

        class FakeResp:
            status_code = 200

            def raise_for_status(self) -> None: ...

            def json(self) -> dict:
                return fake_media

        with patch("paku.web.app.requests.post", return_value=FakeResp()):
            resp = client.post("/api/search/add", json={"anilist_id": 77})
        assert resp.status_code == 200
        body = resp.json()
        assert body["anilist_id"] == 77
        assert body["extraction_mode"] == "manual_search"
        assert body["confidence"] == pytest.approx(1.0)


# --- /api/stats ---


class TestStatsEndpoint:
    def test_stats_empty(self, client: TestClient) -> None:
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["needs_review_count"] == 0

    def test_stats_after_digest(self, client: TestClient, anime_png: bytes) -> None:
        with patch("paku.web.app.process_image") as mock_proc:
            mock_proc.return_value = _fake_anime_result("s1.png", anilist_id=1)
            client.post(
                "/api/digest", files={"file": ("s1.png", anime_png, "image/png")}
            )
            mock_proc.return_value = _fake_anime_result(
                "s2.png", anilist_id=2, title="Other"
            )
            client.post(
                "/api/digest", files={"file": ("s2.png", anime_png, "image/png")}
            )
        body = client.get("/api/stats").json()
        assert body["total"] == 2
        assert "Adventure" in body["by_genre"]


# --- Index / static ---


class TestIndex:
    def test_root_serves_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
