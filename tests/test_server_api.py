from fastapi.testclient import TestClient

from server.app import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body.get("status") in {"ok", "healthy"}


def test_reset_and_step_smoke_wrapped_payload():
    client = TestClient(app)
    reset_resp = client.post("/reset", json={"task": "easy", "seed": 42})
    assert reset_resp.status_code == 200
    reset_body = reset_resp.json()
    assert "observation" in reset_body

    step_resp = client.post(
        "/step",
        json={"action": {"renewable_ratio": 0.6, "fossil_ratio": 0.3, "battery_action": 0.0}},
    )
    assert step_resp.status_code == 200
    step_body = step_resp.json()
    assert "observation" in step_body
    assert "reward" in step_body
    assert "done" in step_body


def test_step_accepts_unwrapped_action_payload():
    client = TestClient(app)
    client.post("/reset", json={"task": "medium", "seed": 7})
    step_resp = client.post(
        "/step",
        json={"renewable_ratio": 0.55, "fossil_ratio": 0.35, "battery_action": 0.0},
    )
    assert step_resp.status_code == 200
    assert "observation" in step_resp.json()


def test_step_invalid_payload_returns_422_not_500():
    client = TestClient(app)
    client.post("/reset", json={"task": "easy", "seed": 11})
    bad_resp = client.post(
        "/step",
        json={"renewable_ratio": 1.5, "fossil_ratio": 1.5, "battery_action": 2.0},
    )
    assert bad_resp.status_code == 422


def test_step_stress_multiple_interactions():
    client = TestClient(app)
    client.post("/reset", json={"task": "hard", "seed": 21})
    for _ in range(20):
        resp = client.post(
            "/step",
            json={"renewable_ratio": 0.6, "fossil_ratio": 0.3, "battery_action": 0.0},
        )
        assert resp.status_code == 200
