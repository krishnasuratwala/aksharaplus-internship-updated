import json
import base64
import time
import os
from google.generativeai import GenerativeModel
import google.generativeai as genai

# Configure Gemini API (replace with your API key)
API_KEY = "AIzaSyDVC_PQ13wxFTeLQxv5-X-2l1iY6UQpin4"  # Get from https://aistudio.google.com/
genai.configure(api_key=API_KEY)

# Initialize Gemini model (Gemini 1.5 Flash for free tier)
model = GenerativeModel("gemini-1.5-flash")

# Load dataset
with open("dataset.json", "r") as f:
    dataset = json.load(f)

# Batch size and delay (to avoid rate limits: 15 RPM)
BATCH_SIZE = 10
DELAY_SECONDS = 50  # 60 seconds delay between batches

# Prompt for Gemini
PROMPT = """
Generate a concise and consistent caption for the scatter plot image. The caption should describe:
1. The type of plot (scatter plot).
2. The number of points per class.
3. The classes and their colors (Class 0: blue, Class 1: red).
4. Whether the classes appear separable (distinct) or overlapping(which can be sepearted using the lm classification algorithems like knn,logistic regression,random forest,decision tree,svm).
5. Keep the tone neutral and factual, avoiding subjective interpretations.
6.just tell the number of points per class and the color of the classes and whether they are separable or not.
Example: "Scatter plot with 10 points per class, showing Class 0 (blue) and Class 1 (red), with distinct separation."
"""

# Process images in batches
total_images = len(dataset)
for batch_start in range(0, total_images, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, total_images)
    print(f"Processing batch {batch_start // BATCH_SIZE + 1}: Images {batch_start} to {batch_end - 1}")
    
    for i in range(batch_start, batch_end):
        entry = dataset[i]
        image_path = entry["image_path"]
        n_points = len(entry["plot_data"]["y"]) // 2  # Number of points per class
        separable = entry["separable"]
        
        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Prepare request for Gemini
        try:
            response = model.generate_content([
                {"text": PROMPT},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_data
                    }
                }
            ])
            
            # Extract caption
            caption = response.text.strip()
            dataset[i]["caption"] = caption
            print(f"Generated caption for image {i}: {caption}")
        
        except Exception as e:
            print(f"Error processing image {i}: {str(e)}")
            dataset[i]["caption"] = "Error: Caption generation failed."
    
    # Save dataset after each batch
    with open("dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved after batch {batch_start // BATCH_SIZE + 1}")
    
    # Delay between batches (except for the last batch)
    if batch_end < total_images:
        print(f"Waiting {DELAY_SECONDS} seconds to avoid rate limits...")
        time.sleep(DELAY_SECONDS)

print("Caption generation complete. All captions added to dataset.json.")