import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

# --- STEP 1: LOAD FILES ---
# Get a list of text files in the current directory
student_files = [file for file in os.listdir() if file.endswith('.txt')]

# Read the content of each file
student_docs = [open(file, encoding='utf-8').read() for file in student_files]

# Print loaded files for verification
print(f"Loaded {len(student_files)} files: {student_files}")


# --- STEP 2: VECTORIZATION (TF-IDF) ---
def create_tfidf_vectors(docs):
    """
    Converts a list of text documents into a TF-IDF matrix.
    Rows = Documents, Columns = Words.
    """
    return TfidfVectorizer().fit_transform(docs).toarray()

# Create vectors for all loaded documents
doc_vec = create_tfidf_vectors(student_docs)

# Pair the filenames with their corresponding vectors
doc_filename_pairs = list(zip(student_files, doc_vec))


# --- STEP 3: COSINE SIMILARITY FUNCTION ---
def calc_cosine_similarity(vector1, vector2):
    """
    Calculates the cosine similarity between two vectors.
    Returns a score between 0 (different) and 1 (identical).
    """
    # reshape needed because cosine_similarity expects 2D arrays
    return cosine_similarity([vector1, vector2])


# --- STEP 4: CHECK PLAGIARISM ---
def find_plagiarism():
    """
    Compares every file against every other file to check for similarity.
    """
    plagiarism_results = set()
    global doc_filename_pairs  # Access the global variable

    # Iterate through each student's file
    for student_a_file, student_a_vec in doc_filename_pairs:
        
        # Create a copy to compare against others
        remaining_pairs = doc_filename_pairs.copy()
        
        # Find index of current student to remove them (don't compare to self)
        current_index = remaining_pairs.index((student_a_file, student_a_vec))
        del remaining_pairs[current_index]

        # Compare with all other students
        for student_b_file, student_b_vec in remaining_pairs:
            
            # Calculate similarity
            similarity_score = calc_cosine_similarity(student_a_vec, student_b_vec)[0][1]
            
            # Sort filenames so (john, juma) is same as (juma, john)
            sorted_filenames = sorted((student_a_file, student_b_file))
            
            # Store result
            result = (sorted_filenames[0], sorted_filenames[1], similarity_score)
            plagiarism_results.add(result)

    return plagiarism_results

# Run the plagiarism checker and print results
print("\n--- Plagiarism Report ---")
results = find_plagiarism()
for result in results:
    print(f"File 1: {result[0]} | File 2: {result[1]} | Similarity: {result[2]:.4f}")


# --- STEP 5: VISUALIZATION (WORD CLOUD) ---
def generate_word_cloud(document_text, filename):
    """
    Generates and shows a word cloud for a specific document.
    """
    wordcloud = WordCloud(width=800, height=400).generate(document_text)
    
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {filename}')
    plt.axis('off')
    plt.show()

# Show Word Cloud for any document with high plagiarism (e.g., > 50%)
print("\n--- Generating Visualizations ---")
for result in results:
    if result[2] >= 0.5:  # Threshold: Only show if similarity is > 0.5
        print(f"High similarity found between {result[0]} and {result[1]}. Generating Word Cloud...")
        
        # Find the text content for the first file in the pair
        file_index = student_files.index(result[0])
        generate_word_cloud(student_docs[file_index], result[0])