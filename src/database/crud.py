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

# Define a function to get a ticket by its ID from the database using SQLAlchemy's session management
def get_ticket_by_id(ticket_id):

    # Use a context manager to create a new session and ensure that the session is properly closed after the operation
    with SessionLocal() as session:

        # Get the ticket with the specified ID from the session
        ticket = session.get(
            Ticket,
            ticket_id
        )
        
        # Return the ticket object
        return ticket

# Test the get_ticket_by_id functions
if __name__ == "__main__":

    # Get the ticket with ID 3 from the database
    ticket = get_ticket_by_id(3)

    # Print the ticket details
    if ticket:
        print("Ticket ID:", ticket.id)
        print("Text:", ticket.text)
        print("Intent:", ticket.predicted_intent)
        print("Confidence:", ticket.confidence)
        print("Created at:", ticket.created_at)
    else:
        # If the ticket is not found, print a message
        print("Ticket not found")