import requests

def test_audio_prediction(audio_file_path, url):
    """
    Sends an audio file to the microservice and prints the response.

    Parameters:
        audio_file_path (str): The path to the audio file.
        url (str): The URL of the microservice endpoint.
    """
    # Open the audio file in binary mode
    with open(audio_file_path, 'rb') as file:
        files = {'file': (audio_file_path, file, 'audio/ogg')}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        print("Prediction received successfully!")
        print(response.json())
    else:
        print("Failed to get prediction.")
        print("Status Code:", response.status_code)
        print("Response:", response.text)

if __name__ == '__main__':
    # URL of the microservice
    # Ensure the IP address and port match the Docker container's exposed settings
    service_url = 'http://127.0.0.1:5000/predict/'
    
    # Path to the audio file you want to analyze
    test_audio_path = './data/soundscape_29201.ogg'

    # Test the audio prediction
    test_audio_prediction(test_audio_path, service_url)

