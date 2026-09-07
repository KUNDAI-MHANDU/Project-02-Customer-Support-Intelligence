import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


# Load environment variables from the .env file
load_dotenv()

# Get the database URL from the environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

# Check if the DATABASE_URL environment variable is set
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

# Create a SQLAlchemy engine and sessionmaker for connecting to the database
engine = create_engine(DATABASE_URL)

# Create a sessionmaker for creating new database sessions
SessionLocal = sessionmaker(
    bind=engine
)

# Test the database connection by executing a simple query to get the current database name
def test_connection():

    # Use a context manager to create a new connection and ensure that the connection is properly closed after the operation
    with engine.connect() as connection:

        # Execute a simple SQL query to get the current database name
        result = connection.execute(
            text("SELECT current_database();")
        )

        # Get the database name from the result of the query and print it to verify that the connection was successful
        database_name = result.scalar_one()
        print("Connected to:", database_name)

# Test the database connection when the script is run directly
if __name__ == "__main__":
    test_connection()