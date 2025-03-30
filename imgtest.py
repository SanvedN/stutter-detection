import requests
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt


def fetch_and_display_result(task_id):
    """Fetch analysis result from API and display the visualization image"""
    try:
        # 1. Make request to your Flask API
        response = requests.get(f"http://localhost:5000/get_result/{task_id}")
        response.raise_for_status()  # Raise error for bad status codes

        result = response.json()

        # 2. Extract and decode the base64 image
        if "visualization" not in result:
            raise ValueError("No visualization found in the response")

        image_data = base64.b64decode(result["visualization"])
        img = Image.open(BytesIO(image_data))

        # 3. Display the image with analysis info
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title(f"Speech Analysis Visualization\nTask ID: {task_id}")
        plt.axis("off")

        # Add analysis summary text
        analysis_text = (
            f"Fluency Score: {result.get('fluency_score', 'N/A')}\n"
            f"Stutter Events: {len(result.get('stutter_events', []))}\n"
            f"Severity: {result.get('severity', 'N/A')}"
        )
        plt.text(
            10,
            30,
            analysis_text,
            fontsize=12,
            color="white",
            bbox=dict(facecolor="black", alpha=0.7),
        )

        plt.show()

        print("✅ Successfully displayed analysis results")
        return result

    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
    except Exception as e:
        print(f"❌ Error processing results: {e}")


# Example usage
if __name__ == "__main__":
    task_id = "20250331_012059"  # Replace with your actual task ID
    fetch_and_display_result(task_id)
