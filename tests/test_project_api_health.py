from fastapi.testclient import TestClient

from backend.main import app


def test_health_endpoint_reports_service():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "DeepShield API"
    assert body["model_status"] in {"not_loaded", "experimental", "unavailable"}


def test_frontend_is_served():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "DeepShield" in response.text


def test_unsupported_video_extension_is_rejected():
    client = TestClient(app)
    response = client.post(
        "/api/v1/detection/video",
        files={"file": ("clip.txt", b"not a video", "text/plain")}
    )
    assert response.status_code == 400
