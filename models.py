"""
SQLAlchemy models for the AeroInsights database.
"""

from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///aeroinsights.db"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class AirportData(Base):
    """Aggregated airport metrics and cluster assignments."""
    __tablename__ = "airport_data"

    ORIGIN_AIRPORT_NAME = Column(String, primary_key=True)
    LATITUDE = Column(Float)
    LONGITUDE = Column(Float)
    AVG_DELAY = Column(Float)
    TOTAL_FLIGHTS = Column(Integer)
    CLUSTER = Column(Integer)


class FlightSample(Base):
    """Sample of individual flight records for dashboard analysis."""
    __tablename__ = "flights_sample"

    flight_id = Column("rowid", Integer, primary_key=True)
    AIRLINE_NAME = Column(String)
    ARRIVAL_DELAY = Column(Float)
    SEASON = Column(String)
    MONTH = Column(Integer)


def init_db():
    """Create all tables in the database."""
    Base.metadata.create_all(engine)


def get_session():
    """Return a new database session."""
    return SessionLocal()
