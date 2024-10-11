import json
from sqlmodel import create_engine_and_session, RpiDevices, SpeciesDetection
from datetime import datetime
from datetime import timedelta


# Create engine and session
engine, session = create_engine_and_session()

def seed_data(json_data):
    if session is None:
        print("Failed to create session. Check your database connection.")
        return
    
    try:
        # Check if a device record already exists for the given pi_id and date
        existing_device = session.query(RpiDevices).filter_by(
            pi_id=json_data['pi_id'],
            analysis_date=datetime.strptime(json_data['date'], '%Y-%m-%d').date()
        ).first()

        if existing_device:
            print(f"Device record already exists for {json_data['pi_id']} on {json_data['date']}. Skipping insertion.")
            return
        
        # Create new RpiDevices record
        new_device = RpiDevices(
            pi_id=json_data['pi_id'],
            analysis_date=datetime.strptime(json_data['date'], '%Y-%m-%d')
        )
        session.add(new_device)
        session.flush()  # Get the id for foreign key use
        
        # Create species detection records
        for detection_dict in json_data['species']:
            for time_segment, detection_data in detection_dict.items():
                # Assuming the time segment format is "HH-MM-SS_dur=600secs_X"
                # We can parse the time from the segment name
                time_str = time_segment.split('_')[0]  # Extracting the HH-MM-SS part
                hour, minute, second = map(int, time_str.split('-'))
                start_time = datetime.strptime(json_data['date'], '%Y-%m-%d') + timedelta(hours=hour, minutes=minute, seconds=second)
                end_time = start_time + timedelta(seconds=5)  # 5 seconds for the detection
                
                species_detection = SpeciesDetection(
                    device_id=new_device.id,
                    time_segment=time_segment,
                    start_time=start_time,
                    end_time=end_time,
                    species_class=detection_data['Class'],
                    confidence_score=detection_data['Score']
                )
                session.add(species_detection)
        
        session.commit()
        print(f"Successfully added data for {new_device.pi_id} on {new_device.analysis_date}")
        
    except Exception as e:
        if session:
            session.rollback()
        print(f"Error seeding data: {str(e)}")
    finally:
        if session:
            session.close()

json_path = "00-07-52_dur=600secs.json"  
with open(json_path, 'r') as file:
    json_data = json.load(file)

seed_data(json_data)