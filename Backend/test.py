import requests
import time

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
        result = response.json()
        print(result)
        return result.get("task_id")
    else:
        print("❌ Upload failed!")
        print(response.text)
        return None


def test_task_status(task_id):
    """Test the /task_status/<task_id> endpoint to check processing status."""
    url = f"{BASE_URL}/task_status/{task_id}"
    print(f"Checking status for task: {task_id}...")

    while True:
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            print(result)
            if result["status"] == "completed" or result["status"] == "failed":
                break
        else:
            print("❌ Failed to check status!")
            print(response.text)
            break

        print("⏳ Still processing... Checking again in 3 seconds.")
        time.sleep(10)


def test_api():
    """Runs the full test sequence for the API."""
    TEST_AUDIO_FILE = "test2.wav"

    # Step 1: Upload Audio
    task_id = test_upload_audio(TEST_AUDIO_FILE)
    if task_id:
        # Step 2: Check Task Status
        test_task_status(task_id)


if __name__ == "__main__":
    test_api()
