import os
import shutil
import json

# Paths (modify these as needed)
source_dir = "/CUB_200_2011/images"
destination_dir = "frontend/public/images"

# Create destination folder if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Load class labels JSON (mapping index to bird name)
with open("../deployment/class_labels.json", "r") as f:
    class_labels = json.load(f)

# Iterate over each class index (assumed to be from "0" to "199")
for idx_str, species in class_labels.items():
    idx = int(idx_str)
    # Folder numbering: index 0 corresponds to folder "001.*"
    folder_num = str(idx + 1).zfill(3)
    folder_name = None

    # Find the folder that starts with the expected folder number
    for folder in os.listdir(source_dir):
        if folder.startswith(folder_num + "."):
            folder_name = folder
            break

    if not folder_name:
        print(f"No folder found for index {idx} ({species})")
        continue

    folder_path = os.path.join(source_dir, folder_name)
    # Get a sorted list of image files in the folder
    files = sorted(os.listdir(folder_path))
    if not files:
        print(f"No images found in folder {folder_name}")
        continue

    # Use the first image in the sorted list
    first_image = files[0]
    source_file_path = os.path.join(folder_path, first_image)
    destination_file_path = os.path.join(destination_dir, f"{idx}.jpg")

    # Copy the file to the destination folder
    shutil.copy(source_file_path, destination_file_path)
    print(f"Copied {source_file_path} to {destination_file_path}")
