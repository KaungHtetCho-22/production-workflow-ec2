# models.py
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()

class AudioAnalysis(Base):
    __tablename__ = 'audio_analysis'
    
    id = Column(Integer, primary_key=True)
    pi_id = Column(String(50), nullable=False)
    analysis_date = Column(Date, nullable=False)
    
    # Relationship to species detections
    detections = relationship("SpeciesDetection", back_populates="analysis")

class SpeciesDetection(Base):
    __tablename__ = 'species_detections'
    
    id = Column(Integer, primary_key=True)
    audio_id = Column(Integer, ForeignKey('audio_analyses.id'), nullable=False)
    time_segment = Column(String(100), nullable=False)
    species_class = Column(String(100), nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Relationship to audio analysis
    analysis = relationship("AudioAnalysis", back_populates="detections")

# Database connection setup
def create_session():
    # Update these values with your actual database credentials
    DATABASE_URL = "mysql+mysqlconnector://biosound_user:biosound_password@localhost/species_db"
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()