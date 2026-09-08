from unittest.mock import patch
from datetime import datetime
from types import SimpleNamespace
from fastapi.testclient import TestClient

from src.api import app


# Create a TestClient instance for the FastAPI app to simulate HTTP requests in the test cases
client = TestClient(app)

# Define a test function to test the /predict endpoint with a valid message
def test_predict_valid_message():

    # Use the patch function from unittest.mock to mock the save_ticket function in src.api, so that it does not actually save tickets to the database during testing
     with patch("src.api.save_ticket") as mock_save_ticket:
        # Define the return value of the mock_save_ticket function to simulate a saved ticket with an ID and created_at timestamp
        mock_save_ticket.return_value.id = 123
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
        assert data["ticket_id"] == 123
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


# Define a test function to test the /tickets endpoint with fake database resultta
def test_get_existing_ticket():


    fake_ticket = SimpleNamespace(
        id=3,
        text="My cash withdrawal is still showing as pending",
        predicted_intent="pending_cash_withdrawal",
        confidence=0.91,
        created_at=datetime.now()
    )

    with patch("src.api.get_ticket_by_id") as mock_get_ticket:

        mock_get_ticket.return_value = fake_ticket

        response = client.get("/tickets/3")

        assert response.status_code == 200

        data = response.json()

        assert data["ticket_id"] == fake_ticket.id
        assert data["text"] == fake_ticket.text
        assert data["predicted_intent"] == fake_ticket.predicted_intent


# Define a test function to test the /tickets endpoint with a missing ticket
def test_get_missing_ticket():

    # Use the patch function from unittest.mock to mock the get_ticket_by_id function in src.api, so that it returns None when the ticket is not found
    with patch("src.api.get_ticket_by_id") as mock_get_ticket:

        # Define the return value of the mock_get_ticket function to be None
        mock_get_ticket.return_value = None

        # Send a GET request to the /tickets endpoint with a missing ticket ID and check the response
        response = client.get("/tickets/99999")

        # Assert that the response status code is 404 (Not Found) since the ticket is not found
        assert response.status_code == 404

        # Get the JSON data from the response and assert that it contains the expected keys and types
        data = response.json()

        # Assert that the response data contains the expected keys and types
        assert data["detail"] == "Ticket not found"

# Define a test function to test the /tickets endpoint with an invalid ticket ID
def test_get_ticket_invalid_id():

    # Send a GET request to the /tickets endpoint with an invalid ticket ID and check the response
    response = client.get("/tickets/not-a-number")

    # Assert that the response status code is 422 (Unprocessable Entity) since the ticket ID is not a valid integer
    assert response.status_code == 422