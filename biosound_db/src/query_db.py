from datetime import datetime

def query_species_by_date(pi_id, date):
    target_date = datetime.strptime(date, '%Y-%m-%d')
    results = session.query(SpeciesDetection).join(RpiDevices).filter(
        RpiDevices.pi_id == pi_id,
        RpiDevices.analysis_date == target_date
    ).all()
    
    for detection in results:
        print(f"Time Segment: {detection.time_segment}, Class: {detection.species_class}, Score: {detection.confidence_score}, Start: {detection.start_time}, End: {detection.end_time}")

# Example usage
query_species_by_date("RPiID-00000000b36010d2", "2024-09-25")
