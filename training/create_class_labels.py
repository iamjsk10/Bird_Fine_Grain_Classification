import os
import json


def create_class_labels(dataset_path, json_file_path):
    class_labels = {}
    species_folders = sorted(os.listdir(dataset_path))

    for index, species in enumerate(species_folders):
        # Skip any non-directory files (like .DS_Store)
        if not os.path.isdir(os.path.join(dataset_path, species)):
            continue

        # Clean the species name by removing leading numbers and underscores
        cleaned_species = species.replace('_', ' ').lstrip('0123456789 ')

        # Add to class_labels dictionary with index starting from 0
        class_labels[str(index)] = cleaned_species

    # Write the class labels to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(class_labels, json_file, indent=4)


# Example usage
DATASET_PATH = "/CUB_200_2011/images"
JSON_FILE_PATH = "../deployment/class_labels.json"

create_class_labels(DATASET_PATH, JSON_FILE_PATH)
print(f"Class labels saved to {JSON_FILE_PATH}")
