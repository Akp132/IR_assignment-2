import numpy as np
import os
import pickle
from scipy.spatial.distance import cosine

def fetch_data(filepath):
    """Load and return data stored in a pickle file."""
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def calculate_similarity(vector_a, vector_b):
    """Calculate the cosine similarity between two vectors or arrays of vectors."""
    if isinstance(vector_a, np.ndarray) and isinstance(vector_b, list):
        vector_b = np.array(vector_b)  # Convert list of arrays to a 2D array
        vector_a_normalized = vector_a / np.linalg.norm(vector_a)
        vector_b_normalized = vector_b / np.linalg.norm(vector_b, axis=1, keepdims=True)
        return np.dot(vector_a_normalized, vector_b_normalized.T)
    elif isinstance(vector_a, dict) and isinstance(vector_b, list):
        sim_scores = []
        for vec in vector_b:
            shared_keys = set(vector_a).intersection(vec)
            dot_prod = sum(vector_a[k] * vec[k] for k in shared_keys)
            norm_a = np.sqrt(sum(v**2 for v in vector_a.values()))
            norm_b = np.sqrt(sum(v**2 for v in vec.values()))
            sim = dot_prod / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0
            sim_scores.append(sim)
        return np.array(sim_scores)
    else:
        raise TypeError("Invalid vector types for similarity calculation.")

def retrieve_similars(data, query_feature, n_top=3):
    """Identify the top N similar items to a query feature based on cosine similarity."""
    sim_scores = calculate_similarity(query_feature, data)
    top_indices = np.argsort(sim_scores)[-n_top:][::-1]
    return top_indices, sim_scores[top_indices]

# Define file paths
data_dir = '/path/to/data'
features_file = os.path.join(data_dir, 'image_features.pkl')
tfidf_file = os.path.join(data_dir, 'tfidf.pkl')

# Load the features and TF-IDF scores
features = fetch_data(features_file)
tfidf_scores = fetch_data(tfidf_file)

# Example query
query_feature_img = features[0]  # First feature as example
query_feature_text = tfidf_scores[0]  # First TF-IDF score as example

# Find similar items
img_matches, img_similarities = retrieve_similars(features, query_feature_img)
text_matches, text_similarities = retrieve_similars(tfidf_scores, query_feature_text, n_top=3)

# Output the similarity results
print("Matching Image Indices:", img_matches)
print("Image Similarities:", img_similarities)
print("Matching Text Indices:", text_matches)
print("Text Similarities:", text_similarities)

# Save the results
results_save_path = os.path.join(data_dir, 'match_results.pkl')
with open(results_save_path, 'wb') as res_file:
    pickle.dump({
        'img_indices': img_matches,
        'img_sims': img_similarities,
        'text_indices': text_matches,
        'text_sims': text_similarities,
    }, res_file)
