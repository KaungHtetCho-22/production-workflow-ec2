# seed_data.py
import json
from model import create_session, AudioAnalysis, SpeciesDetection
from datetime import datetime

def seed_data(json_data):
    session = create_session()
    
    try:
        # Create audio analysis record
        audio_analysis = AudioAnalysis(
            pi_id=json_data['pi_id'],
            analysis_date=datetime.strptime(json_data['date'], '%Y-%m-%d').date()
        )
        session.add(audio_analysis)
        session.flush()  # This ensures we get the id of the audio_analysis
        
        # Create species detection records
        for detection_dict in json_data['species']:
            for time_segment, detection_data in detection_dict.items():
                species_detection = SpeciesDetection(
                    audio_id=audio_analysis.id,
                    time_segment=time_segment,
                    species_class=detection_data['Class'],
                    confidence_score=detection_data['Score']
                )
                session.add(species_detection)
        
        session.commit()
        print(f"Successfully added data for {audio_analysis.pi_id} on {audio_analysis.analysis_date}")
        
    except Exception as e:
        session.rollback()
        print(f"Error seeding data: {str(e)}")
    finally:
        session.close()

# Example usage
json_data = {
    "pi_id": "RPiID-00000000b36010d2",
    "date": "2024-09-25",
    "species": [{
        "00-07-52_dur=600secs_5": {
            "Class": "Centropus-sinensis",
            "Score": 0.14264535903930664
        },
        "00-07-52_dur=600secs_10": {
            "Class": "Centropus-sinensis",
            "Score": 0.1116638258099556
        }
    }]
}

seed_data(json_data)