from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class TestTable(Base):
    __tablename__ = 'test_table'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))

def create_session():
    try:
        DATABASE_URL = "mysql+mysqlconnector://biosound_user:biosound_password@localhost/species_db"
        engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(engine)  # Create tables
        Session = sessionmaker(bind=engine)
        print("Connected and created test_table.")
    except Exception as e:
        print(f"Error: {e}")

create_session()
