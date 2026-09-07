from datetime import datetime

from sqlalchemy import BigInteger, String, Text, Float, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# Define the base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass


# Define the Ticket model for the tickets table in the database, which includes columns for id, text, predicted_intent, confidence, and created_at
class Ticket(Base):
    # Define the table name for the tickets table
    __tablename__ = "tickets"

    # Define the columns for the tickets table using SQLAlchemy's mapped_column and Mapped types
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    predicted_intent: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

