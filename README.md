
Bird Classifier Backend
------------------------------------------------------------

Overview:
-----------
This repository contains the backend API for a bird classification project. The API is built using Flask and TensorFlow. It uses an EfficientNet-based model pre-trained on the CUB-200-2011 dataset to classify bird species. The API exposes a /predict endpoint that accepts an image file and returns the predicted species along with relevant metadata (name, habitat, diet, wingspan). Note that the frontend (a Pokédex-inspired UI) is maintained in a separate repository, and screenshots/videos from the frontend are available there.

How It Works:
---------------
1. A POST request is sent to the /predict endpoint with an image file (accepted formats: PNG, JPG, JPEG, GIF).
2. The API validates the file and preprocesses the image (resizing to 224x224 and applying EfficientNet preprocessing).
3. The pre-trained model performs inference on the image.
4. The API returns a JSON response containing:
   - species: An object with metadata (name, habitat, diet, wingspan)
   - species_id: The index of the predicted class as a string
   - confidence: The prediction confidence as a float

Key Files:
-----------
- backend.py : Contains the Flask application and route definitions.
- recovery_model.keras: The pre-trained EfficientNet model fine-tuned on the CUB-200-2011 dataset.
- class_labels.json: A JSON file mapping each class index to its corresponding species information.

Example Response:
------------------
{
  "species": {
    "name": "Western Meadowlark",
    "habitat": "Grasslands and prairies",
    "wingspan": "36-40 cm",
    "diet": "Insects, seeds"
  },
  "species_id": "0",
  "confidence": 0.9976
}

Setup & Running Locally:
---------------------------
1. Clone the repository.
2. Install the required dependencies using:
   pip install -r requirements.txt
3. Run the Flask application:
   python backend.py
   The app will listen on http://127.0.0.1:5001 (or as configured).

Screenshots & Demo:
--------------------
- Although this repository is only for the backend, screenshots and videos _(in demo folder)_ demonstrating the complete application (including the frontend) are available used to demonstrate.
![Demo Screenshot](https://raw.githubusercontent.com/iamjsk10/Bird_Fine_Grain_Classification/main/Demo/Screenshot%202025-03-26%20at%203.57.51%E2%80%AFPM.png)
![Demo Screenshot](https://raw.githubusercontent.com/iamjsk10/Bird_Fine_Grain_Classification/main/Demo/Screenshot%202025-03-26%20at%203.59.12%E2%80%AFPM.png)


Notes:
-------
- The model uses the CUB-200-2011 dataset and an EfficientNet-based architecture. If you change or update the model, make sure the input size and class mapping in class_labels.json remain consistent.
- This backend is designed to work seamlessly with the corresponding frontend interface.

License:
---------
This project is licensed under the MIT License. Feel free to modify and distribute as needed.

