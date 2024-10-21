import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship


APP_DATA_DIR = os.getenv('APP_DATA_DIR')
if not APP_DATA_DIR:
    raise Exception("Environment variable APP_DATA_DIR not set.")
if not os.path.isdir(APP_DATA_DIR):
    raise Exception(f"Directory {APP_DATA_DIR} not found.")
if APP_DATA_DIR.endswith('/'):
    APP_DATA_DIR = APP_DATA_DIR[:-1]

DATABASE_URL = f'sqlite:///{APP_DATA_DIR}/sql_app.db'
engine = create_engine(DATABASE_URL)
create_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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


Base.metadata.create_all(bind=engine)
