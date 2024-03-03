import numpy as np
import pickle
import os
from scipy.spatial.distance import cdist

def load_data(file_path):
    """Utility function to load data from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors or sets of vectors."""
    if isinstance(v1, np.ndarray) and isinstance(v2, list) and all(isinstance(v, np.ndarray) for v in v2):
        # Case for dense vectors (image features)
        v2 = np.vstack(v2)  # Stack list of numpy arrays into a single 2D numpy array
        # Normalize vectors to unit length
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
        # Compute cosine similarity
        return np.dot(v1_norm, v2_norm.T)
    elif isinstance(v1, dict) and isinstance(v2, list) and all(isinstance(v, dict) for v in v2):
        # Case for sparse vectors (TF-IDF scores)
        similarities = []
        for v2_dict in v2:
            # Intersection of keys (terms present in both vectors)
            common_terms = set(v1.keys()) & set(v2_dict.keys())
            # Manual dot product for common terms
            dot_product = sum(v1[term] * v2_dict[term] for term in common_terms)
            # Norms of the vectors
            norm_v1 = np.sqrt(sum(value ** 2 for value in v1.values()))
            norm_v2 = np.sqrt(sum(value ** 2 for value in v2_dict.values()))
            # Cosine similarity
            if norm_v1 == 0 or norm_v2 == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_v1 * norm_v2)
            similarities.append(similarity)
        return np.array(similarities)
    else:
        raise ValueError("Unsupported input types for similarity computation.")


def find_most_similar(features, input_feature, top_n=3):
    """Find the most similar items based on cosine similarity."""
    similarities = compute_cosine_similarity(input_feature, features)
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return top_indices, similarities[top_indices]

# Paths to the data
output_dir = '/Users/akshay/Desktop/IR/Assinment2/output'
image_features_path = os.path.join(output_dir, 'image_features.pkl')
tf_idf_scores_path = os.path.join(output_dir, 'tf_idf_results.pkl')

# Loading preprocessed data
image_features = load_data(image_features_path)
tf_idf_scores = load_data(tf_idf_scores_path)

# Process and find similar images and reviews for an example input
example_image_feature = image_features[0]  # Using the first image feature as an example input
example_review_feature = tf_idf_scores[0]  # Using the first TF-IDF score as an example input

similar_image_indices, image_similarities = find_most_similar(image_features, example_image_feature)
similar_review_indices, review_similarities = find_most_similar(tf_idf_scores, example_review_feature, top_n=3)

# Displaying results
print("Similar Image Indices:", similar_image_indices)
print("Image Similarities:", image_similarities)
print("Similar Review Indices:", similar_review_indices)
print("Review Similarities:", review_similarities)

# Saving retrieval results
retrieval_results = {
    'similar_image_indices': similar_image_indices,
    'image_similarities': image_similarities,
    'similar_review_indices': similar_review_indices,
    'review_similarities': review_similarities,
}

results_path = os.path.join(output_dir, 'retrieval_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(retrieval_results, f)