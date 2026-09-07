from src.database.database import SessionLocal
from src.database.models import Ticket


# Define a function to save a ticket to the database using SQLAlchemy's session management
def save_ticket(text, predicted_intent, confidence):

    # Use a context manager to create a new session and ensure that the session is properly closed after the operation
    with SessionLocal() as session:

        # Create a new Ticket object with the provided text, predicted intent, and confidence
        ticket = Ticket(
            text= text,
            predicted_intent= predicted_intent,
            confidence= confidence
        )

        # Add the new ticket to the session
        session.add(ticket)

        # Commit the transaction to save the ticket to the database
        session.commit()

        # Refresh the ticket to get its ID and created_at timestamp after committing the transaction
        session.refresh(ticket)

        # Return the saved ticket object, which now contains the ID and created_at timestamp
        return ticket

# Test the save_ticket function by saving a sample ticket to the database and printing its ID and created_at timestamp
if __name__ == "__main__":
    # Save a sample ticket to the database using the save_ticket function
    ticket = save_ticket(
        text="The ATM did not give me my cash",
        predicted_intent="cash_withdrawal",
        confidence=0.87
    )

    # Print the ID and created_at timestamp of the saved ticket to verify that it was successfully saved to the database
    print("Ticket ID:", ticket.id)
    print("Created at:", ticket.created_at)