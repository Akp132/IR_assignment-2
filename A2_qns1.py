import torch
from torchvision import models, transforms
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import pickle
import numpy as np
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Configure the directory for saving output
save_directory = '/Users/akshay/Desktop/IR/Assinment2/output'

# Define the image preprocessing steps
image_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize pre-trained ResNet50 model for feature extraction
model_resnet50 = models.resnet50(pretrained=True)
model_resnet50.eval()

# Extract features from a given image URL
def extract_features_from_image(url_image):
    try:
        # Convert list-like string URL to actual list and get the first element
        if url_image.startswith("[") and url_image.endswith("]"):
            url_image = ast.literal_eval(url_image)[0]

        image_response = requests.get(url_image)
        image = Image.open(BytesIO(image_response.content))
        image_transformed = image_preprocess(image)
        image_transformed = image_transformed.unsqueeze(0)  # Add a dimension for batch processing
        with torch.no_grad():
            features_output = model_resnet50(image_transformed)
        return features_output.cpu().numpy().flatten()
    except Exception as error:
        print(f"Failed to process image URL {url_image}: {error}")
        return None

# Clean and preprocess text data
set_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text_review):
    if not isinstance(text_review, str):
        return ""  # Handle cases where input is not a string
    text_lower = text_review.lower()
    text_no_punctuation = ''.join(character for character in text_lower if character not in string.punctuation)
    words_filtered = text_no_punctuation.split()
    words_lemmatized = [lemmatizer.lemmatize(word) for word in words_filtered if word not in set_stopwords]
    return ' '.join(words_lemmatized)

# Load and process dataset
path_dataset = '/Users/akshay/Desktop/IR/Assinment2/A2_Data.csv'
df = pd.read_csv(path_dataset)

# Feature extraction for images
features_image_extracted = []

for _, row in df.iterrows():
    features_img = extract_features_from_image(row['Image'])
    if features_img is not None:
        features_image_extracted.append(features_img)

# Save the extracted features
path_save_features = os.path.join(save_directory, 'image_features.pkl')
with open(path_save_features, 'wb') as file:
    pickle.dump(features_image_extracted, file)

print(f"Image features saved to {path_save_features}")
