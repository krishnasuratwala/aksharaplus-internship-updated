import json
import os
import torch
from torch.utils.data import Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration, Trainer, TrainingArguments
from PIL import Image
from tqdm import tqdm

# Step 1: Verify scatter_plots folder and dataset.json
scatter_plots_dir = "scatter_plots"
dataset_file = "dataset.json"

# Check if scatter_plots folder exists
if not os.path.exists(scatter_plots_dir):
    raise FileNotFoundError(f"Scatter plots directory not found at: {scatter_plots_dir}")

# Check if dataset.json exists
if not os.path.exists(dataset_file):
    raise FileNotFoundError(f"dataset.json not found at: {dataset_file}")

# Print folder structure to debug
print(f"Listing files in {scatter_plots_dir}:")
for root, dirs, files in os.walk(scatter_plots_dir):
    for file in files:
        print(os.path.join(root, file))

# Step 2: Load dataset
with open(dataset_file, "r") as f:
    dataset = json.load(f)

# Print a few image paths from dataset.json to debug
print("Sample image paths from dataset.json:")
for i in range(min(5, len(dataset))):
    print(dataset[i]["image_path"])

# Step 3: Custom Dataset class
class ScatterPlotCaptionDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # Extract the filename from image_path and construct the correct path
        # dataset.json has image_path like "scatter_plots/plot_0.png"
        # Files are in ./scatter_plots/
        image_filename = os.path.basename(entry["image_path"])  # e.g., "plot_0.png"
        image_path = os.path.join(scatter_plots_dir, image_filename)

        # Debug: Print the constructed path
        if idx < 5:  # Print for the first 5 items
            print(f"Attempting to load image: {image_path}")

        # Verify file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        caption = entry["caption"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Process image and caption
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=50,
            return_tensors="pt"
        )

        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = encoding["input_ids"]

        return encoding

# Step 4: Initialize BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Use CPU
device = torch.device("cpu")
model.to(device)
print(f"Using device: {device}")

# Step 5: Prepare dataset
train_dataset = ScatterPlotCaptionDataset(dataset, processor)

# Step 6: Define training arguments (disable W&B)
training_args = TrainingArguments(
    output_dir="./blip_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    fp16=False,  # Not supported on CPU
    dataloader_num_workers=0,  # Set to 0 for CPU to avoid multiprocessing issues
    remove_unused_columns=False,
    report_to="none",  # Disable W&B logging
)

# Step 7: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Step 8: Fine-tune the model
print("Starting fine-tuning BLIP...")
trainer.train()

# Step 9: Save the fine-tuned model locally
output_dir = "./blip_finetuned"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"Fine-tuned model saved to: {output_dir}")