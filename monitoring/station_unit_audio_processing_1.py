from fastapi import FastAPI, UploadFile, BackgroundTasks
import os
import requests
import json
import logging
from collections import defaultdict
from datetime import datetime
import shutil

# Set up logging
log_dir = '../../logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'processing.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = FastAPI()

# Configuration
service_url = 'http://localhost:5000/predict/'
base_folder = '/home/work/Input/'
output_folder = '/home/work/Output/'
processed_files_path = 'processed_files.txt'

def test_audio_prediction(audio_file_path, url, output_folder, pi_id, date, processed_files):
    """
    Sends an audio file to the microservice, prints the response, and saves it as a JSON file.
    """
    start_time = datetime.now()
    
    # Open the audio file in binary mode
    with open(audio_file_path, 'rb') as file:
        files = {'file': (os.path.basename(audio_file_path), file, 'audio/ogg')}
        response = requests.post(url, files=files)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    if response.status_code == 200:
        result = response.json()
        logging.info(f"File processed: {os.path.basename(audio_file_path)}")
        logging.info(f"Processing started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Processing ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Time taken: {duration:.2f} seconds")
        
        date_output_folder = os.path.join(output_folder, pi_id, date)
        os.makedirs(date_output_folder, exist_ok=True)

        # Save result as JSON with audio file name
        audio_file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        individual_output_file_path = os.path.join(date_output_folder, f'{audio_file_name}.json')
        
        # Load existing data if the file already exists
        if os.path.exists(individual_output_file_path):
            with open(individual_output_file_path, 'r') as infile:
                existing_data = json.load(infile)
        else:
            existing_data = {"pi_id": pi_id, "date": date, "species": []}
        
        # Append the new result
        existing_data["species"].append(result)
        
        # Save individual result
        with open(individual_output_file_path, 'w') as outfile:
            json.dump(existing_data, outfile)
        
        logging.info(f"Result saved: {individual_output_file_path}")
        
        # Update processed_files.txt with finish time
        finish_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
        with open(processed_files, 'a') as f:
            f.write(f"{audio_file_path},{finish_time}\n")
        
    else:
        logging.error(f"Failed to get prediction for {os.path.basename(audio_file_path)}. Status Code: {response.status_code}. Response: {response.text}")

def process_all_sites(base_folder, service_url, output_folder, processed_files_path):
    """
    Process all audio files from the base folder.
    """
    pi_dirs = [d for d in os.listdir(base_folder) if d.startswith('RPiID-')]
    site_files = defaultdict(list)

    # Collect all files in order
    for pi_dir in pi_dirs:
        pi_path = os.path.join(base_folder, pi_dir)
        for date_dir in os.listdir(pi_path):
            date_path = os.path.join(pi_path, date_dir)
            for file in os.listdir(date_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(date_path, file)
                    site_files[pi_dir].append((file_path, date_dir))
    
    # Sort files by date and filename before sending
    for pi_id in site_files:
        site_files[pi_id].sort(key=lambda x: (x[1], x[0]))

    files_processed = False
    # Process files alternately from each site
    while any(site_files.values()):
        for pi_id in site_files:
            if site_files[pi_id]:
                audio_file, date = site_files[pi_id].pop(0)
                logging.info(f"Processing file from {pi_id} on {date}: {os.path.basename(audio_file)}")
                test_audio_prediction(audio_file, service_url, output_folder, pi_id, date, processed_files_path)
                files_processed = True
    
    return files_processed

def load_processed_files(file_path):
    """
    Load the set of processed files from a text file.
    """
    processed_files = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                # Ensure that lines are correctly split
                if ',' in line:
                    file_path, finish_time = line.strip().split(',', 1)
                    processed_files[file_path] = finish_time
                else:
                    logging.error(f"Invalid line in processed_files.txt: {line.strip()}")
    return processed_files

def save_processed_files(file_path, processed_files):
    """
    Save the set of processed files to a text file.
    """
    with open(file_path, 'w') as f:
        for file_path, finish_time in processed_files.items():
            f.write(f"{file_path},{finish_time}\n")

@app.post("/upload/")
async def upload_audio_file(file: UploadFile, background_tasks: BackgroundTasks):
    """
    Endpoint to receive an audio file and process it in the background.
    """
    file_path = os.path.join(base_folder, file.filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the file in the background
    background_tasks.add_task(process_all_sites, base_folder, service_url, output_folder, processed_files_path)
    
    return {"message": f"File {file.filename} uploaded and processing started."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)