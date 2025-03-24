import requests

# Base URL of the Flask API
BASE_URL = "http://127.0.0.1:5000"


def test_upload_audio(file_path):
    """Test the /upload_audio endpoint by uploading an audio file."""
    url = f"{BASE_URL}/upload_audio"
    files = {"file": open(file_path, "rb")}

    print(f"Uploading {file_path}...")
    response = requests.post(url, files=files)

    if response.status_code == 200:
        print("✅ Upload successful!")
        print(response.json())
    else:
        print("❌ Upload failed!")
        print(response.text)


def test_get_results():
    """Test the /get_results endpoint to fetch processed audio results."""
    url = f"{BASE_URL}/get_results"

    print("Fetching analysis results...")
    response = requests.get(url)

    if response.status_code == 200:
        print("✅ Fetched results successfully!")
        print(response.json())
    else:
        print("❌ Failed to fetch results!")
        print(response.text)


if __name__ == "__main__":
    # Specify your test audio file here
    TEST_AUDIO_FILE = "test2.wav"

    # Run tests
    test_upload_audio(TEST_AUDIO_FILE)
    test_get_results()
