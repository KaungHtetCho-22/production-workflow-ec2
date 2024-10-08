#!/bin/bash

# Configuration
SERVICE_URL="http://localhost:5000/predict/"
BASE_FOLDER="/home/work/Input/"
OUTPUT_FOLDER="/home/work/Output/"
PROCESSED_FILES_PATH="processed_files.txt"
LOG_DIR="../../logs"
LOG_FILE="$LOG_DIR/processing.log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Log function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Function to send audio file to the microservice
test_audio_prediction() {
    local audio_file_path="$1"
    local pi_id="$2"
    local date="$3"

    start_time=$(date +%s)

    # Send the audio file to the service
    response=$(curl -s -w "%{http_code}" -F "file=@$audio_file_path" "$SERVICE_URL")
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if [ "$response" -eq 200 ]; then
        result=$(curl -s -F "file=@$audio_file_path" "$SERVICE_URL")
        log "File processed: $(basename "$audio_file_path")"
        log "Processing started at: $(date -d @"$start_time" '+%Y-%m-%d %H:%M:%S')"
        log "Processing ended at: $(date -d @"$end_time" '+%Y-%m-%d %H:%M:%S')"
        log "Time taken: $duration seconds"

        date_output_folder="$OUTPUT_FOLDER/$pi_id/$date"
        mkdir -p "$date_output_folder"

        audio_file_name="${audio_file_path##*/}"
        audio_file_name_without_ext="${audio_file_name%.*}"
        individual_output_file_path="$date_output_folder/$audio_file_name_without_ext.json"

        # Load existing data if the file already exists
        if [ -f "$individual_output_file_path" ]; then
            existing_data=$(<"$individual_output_file_path")
            updated_data=$(echo "$existing_data" | jq --argjson new_result "$result" '.species += [$new_result]')
        else
            updated_data="{\"pi_id\": \"$pi_id\", \"date\": \"$date\", \"species\": [$result]}"
        fi

        echo "$updated_data" > "$individual_output_file_path"
        log "Result saved: $individual_output_file_path"

        finish_time=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$audio_file_path,$finish_time" >> "$PROCESSED_FILES_PATH"
    else
        log "Failed to get prediction for $(basename "$audio_file_path"). Status Code: $response."
    fi
}

# Function to process all audio files
process_all_sites() {
    for pi_dir in "$BASE_FOLDER"/RPiID-*; do
        [ -d "$pi_dir" ] || continue
        pi_id=$(basename "$pi_dir")

        for date_dir in "$pi_dir"/*; do
            [ -d "$date_dir" ] || continue
            date=$(basename "$date_dir")

            for audio_file in "$date_dir"/*.wav; do
                [ -f "$audio_file" ] || continue
                log "Processing file from $pi_id on $date: $(basename "$audio_file")"
                test_audio_prediction "$audio_file" "$pi_id" "$date"
            done
        done
    done
}

# Function to load processed files
load_processed_files() {
    if [ -f "$PROCESSED_FILES_PATH" ]; then
        while IFS=',' read -r file_path finish_time; do
            processed_files["$file_path"]="$finish_time"
        done < "$PROCESSED_FILES_PATH"
    fi
}

# Function to save processed files
save_processed_files() {
    for file_path in "${!processed_files[@]}"; do
        echo "$file_path,${processed_files[$file_path]}" >> "$PROCESSED_FILES_PATH"
    done
}

# Main script execution
if [[ $# -gt 0 ]]; then
    for file in "$@"; do
        # Save uploaded file
        cp "$file" "$BASE_FOLDER/"
        log "File $(basename "$file") uploaded and processing started."

        # Process all audio files
        process_all_sites
    done
else
    echo "No audio files provided. Usage: ./your_script.sh <audio_file1.wav> <audio_file2.wav> ..."
fi
