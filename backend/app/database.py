"""
Revalance — Database Connection
================================
This module sets up the connection between our Python code and PostgreSQL.

KEY CONCEPT — ORM (Object-Relational Mapping):
Instead of writing raw SQL queries like:
    SELECT * FROM zones WHERE zone_id = 42
We can write Python code like:
    session.query(Zone).filter(Zone.zone_id == 42).first()

SQLAlchemy is the ORM library that makes this possible. Think of it as
a translator between Python objects and database tables.

KEY CONCEPT — Session:
A "session" is like a conversation with the database. You open one,
do your queries/inserts, and then close it. We use a pattern called
"dependency injection" in FastAPI to create sessions per-request.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

# Create the database engine — this is the "connection pool"
# A connection pool is like having a team of phone lines to the database
# instead of dialing a new number every time you want to talk
engine = create_engine(
    settings.database_url,
    pool_size=10,       # Keep 10 connections ready
    max_overflow=20,    # Allow 20 extra connections during peak times
    echo=False,         # Set to True to see SQL queries in the console (debugging)
)

# SessionLocal is a "factory" that creates new database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Base class for all our database models (tables)
class Base(DeclarativeBase):
    pass


def get_db():
    """
    FastAPI dependency that provides a database session.
    
    Usage in an endpoint:
        @app.get("/zones")
        def get_zones(db: Session = Depends(get_db)):
            return db.query(Zone).all()
    
    The 'yield' keyword makes this a generator — it gives you the session,
    waits for you to finish, then closes it automatically. This prevents
    "connection leaks" (forgetting to close connections).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
