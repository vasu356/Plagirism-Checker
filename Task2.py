import os
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    return text

def calculate_cosine_similarity(doc1, doc2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix[0][1]

def check_plagiarism():
    results = set()
    global sample_files

    for i in range(len(sample_files)):
        for j in range(i+1, len(sample_files)):
            sample_a = sample_files[i]
            sample_b = sample_files[j]

            with open(sample_a, 'r') as file:
                text_vector_a = preprocess_text(file.read())

            with open(sample_b, 'r') as file:
                text_vector_b = preprocess_text(file.read())

            sim_score = calculate_cosine_similarity(text_vector_a, text_vector_b)

            print("Similarity score between", sample_a, "and", sample_b, ":", sim_score*100,"%")

            if sim_score > 0.8:
                score = (sample_a, sample_b, sim_score)
                results.add(score)

    return results

sample_files = [doc for doc in os.listdir() if doc.endswith('.txt')]

print("Sample Files:", sample_files)

for data in check_plagiarism():
    print("Plagiarism detected between", data[0], "and", data[1], "with a similarity score of ", data[2])
