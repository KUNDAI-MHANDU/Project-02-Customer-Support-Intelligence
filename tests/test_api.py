from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)

def test_predict_valid_message():

    response = client.post(
        "/predict",
        json={
            "text": "My card has not arrived"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "intent" in data
    assert "confidence" in data

    assert isinstance(data["intent"], str)
    assert isinstance(data["confidence"], float)


def test_empty_message():
    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 422


def test_whitespace_message():
    response = client.post(
        "/predict",
        json={"text": "     "}
    )

    assert response.status_code == 422


def test_message_too_long():
    long_message = "a" * 1001

    response = client.post(
        "/predict",
        json={"text": long_message}
    )

    assert response.status_code == 422