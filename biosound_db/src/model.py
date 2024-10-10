from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()

class RpiDevices(Base):
    __tablename__ = 'RpiDevices'
    
    id = Column(Integer, primary_key=True)
    pi_id = Column(String(50), nullable=False)
    analysis_date = Column(DateTime, nullable=False)  # Change to DateTime for time tracking
    
    # Relationship to species detections
    detections = relationship("SpeciesDetection", back_populates="device", cascade="all, delete-orphan")

class SpeciesDetection(Base):
    __tablename__ = 'SpeciesDetection'
    
    id = Column(Integer, primary_key=True)
    device_id = Column(Integer, ForeignKey('RpiDevices.id'), nullable=False)
    time_segment = Column(String(100), nullable=False)
    start_time = Column(DateTime, nullable=False)  # New field for the start time
    end_time = Column(DateTime, nullable=False)    # New field for the end time
    species_class = Column(String(100), nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Relationship to RpiDevices
    device = relationship("RpiDevices", back_populates="detections")

# Database connection setup
def create_engine_and_session():
    DATABASE_URL = "mysql+mysqlconnector://biosound_user:biosound_password@localhost/species_db"
    engine = create_engine(DATABASE_URL, echo=True)
    Session = sessionmaker(bind=engine)
    return engine, Session()

def initialize_database(engine):
    Base.metadata.drop_all(engine)  # This will drop all tables
    Base.metadata.create_all(engine)  # This will create the tables again
    print("Connected and created tables.")


# Create engine and session
engine, session = create_engine_and_session()
initialize_database(engine)
