import pandas as pd
import numpy as np
import math
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Ensure necessary NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextHandler:
    def __init__(self):
        self.forbidden_words = set(stopwords.words('english'))
        self.shortener = PorterStemmer()
        self.normalizer = WordNetLemmatizer()
        # Compile patterns for cleaning
        self.pattern_url = re.compile(r'https?://\S+|www\.\S+')
        self.pattern_social = re.compile(r'@\w+|#\w+')
    
    def purify_text(self, content):
        content = content.lower()
        content = self.pattern_url.sub('', content)
        content = self.pattern_social.sub('', content)
        return content
    
    def segment_text(self, content):
        pieces = word_tokenize(content)
        pieces = [piece for piece in pieces if piece.isalpha()]
        return pieces
    
    def filter_and_refine(self, pieces):
        refined_pieces = [self.shortener.stem(self.normalizer.lemmatize(word))
                          for word in pieces if word not in self.forbidden_words]
        return refined_pieces
    
    def process_text(self, content):
        content = self.purify_text(content)
        pieces = self.segment_text(content)
        final_pieces = self.filter_and_refine(pieces)
        return final_pieces

# Set up the text processor
text_engine = TextHandler()

# Read the dataset
path_to_dataset = '/Users/akshay/Desktop/IR/Assinment2/A2_Data.csv'
df = pd.read_csv(path_to_dataset)

# Clean and process text data
df['Cleaned_Text'] = df['Review Text'].fillna('').apply(lambda x: ' '.join(text_engine.process_text(x)))

# Calculate TF-IDF values
def calculate_tf_idf(documents):
    all_words = sum(documents, [])
    word_df = {word: all_words.count(word) for word in set(all_words)}
    
    total_docs = len(documents)
    idf_values = {word: math.log(total_docs / (word_df[word] + 1)) for word in word_df}
    
    tf_idf_results = []
    for doc in documents:
        word_count = len(doc)
        tf_values = {word: doc.count(word) / word_count for word in set(doc)}
        tf_idf = {word: (tf_values[word] * idf_values[word]) for word in tf_values}
        tf_idf_results.append(tf_idf)
    
    return tf_idf_results

# Process and split the texts for TF-IDF calculation
processed_texts = df['Cleaned_Text'].apply(lambda x: x.split()).tolist()
tf_idf_results = calculate_tf_idf(processed_texts)

# Save processing results
result_folder = '/Users/akshay/Desktop/IR/Assinment2/output'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

with open(os.path.join(result_folder, 'cleaned_texts.pkl'), 'wb') as file:
    pickle.dump(processed_texts, file)

with open(os.path.join(result_folder, 'tf_idf_results.pkl'), 'wb') as file:
    pickle.dump(tf_idf_results, file)
