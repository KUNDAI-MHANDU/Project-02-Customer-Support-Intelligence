from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app


# Create a TestClient instance for the FastAPI app to simulate HTTP requests in the test cases
client = TestClient(app)

# Define a test function to test the /predict endpoint with a valid message
def test_predict_valid_message():

    # Use the patch function from unittest.mock to mock the save_ticket function in src.api, so that it does not actually save tickets to the database during testing
     with patch("src.api.save_ticket") as mock_save_ticket:

        # Send a POST request to the /predict endpoint with a valid message and check the response
        response = client.post(
            "/predict",
            json={
                "text": "My card has not arrived"
            }
        )

        # Assert that the response status code is 200 (OK) and that the response contains the expected keys and types
        assert response.status_code == 200

        # Get the JSON data from the response and assert that it contains the expected keys and types
        data = response.json()

        # Assert that the response data contains the expected keys and types
        assert "intent" in data
        assert "confidence" in data

        # Assert that the intent is a string and the confidence is a float
        assert isinstance(data["intent"], str)
        assert isinstance(data["confidence"], float)

        # Assert that the mock_save_ticket function was called once with the expected arguments, including the text, predicted intent, and confidence
        mock_save_ticket.assert_called_once_with(
            text="My card has not arrived",
            predicted_intent=data["intent"],
            confidence=data["confidence"]
        )


# Define a test function to test the /predict endpoint with an empty message, which should return a 422 Unprocessable Entity status code
def test_empty_message():
    # Send a POST request to the /predict endpoint with an empty message and check the response
    response = client.post(
        "/predict",
        json={"text": ""}
    )

    # Assert that the response status code is 422 (Unprocessable Entity) since the message is empty
    assert response.status_code == 422


# Define a test function to test the /predict endpoint with a whitespace-only message, which should return a 422 Unprocessable Entity status code
def test_whitespace_message():
    # Send a POST request to the /predict endpoint with a whitespace-only message and check the response
    response = client.post(
        "/predict",
        json={"text": "     "}
    )

    # Assert that the response status code is 422 (Unprocessable Entity) since the message is only whitespace
    assert response.status_code == 422


# Define a test function to test the /predict endpoint with a message that exceeds the maximum length of 1000 characters, which should return a 422 Unprocessable Entity status code
def test_message_too_long():
    # Create a message that exceeds the maximum length of 1000 characters by repeating the character "a" 1001 times
    long_message = "a" * 1001

    # Send a POST request to the /predict endpoint with the long message and check the response
    response = client.post(
        "/predict",
        json={"text": long_message}
    )

    # Assert that the response status code is 422 (Unprocessable Entity) since the message exceeds the maximum length
    assert response.status_code == 422