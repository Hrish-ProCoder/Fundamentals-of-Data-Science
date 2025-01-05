import time
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(input_file, output_file, user_threshold=5):
    """
    Optimized function to compute similarity scores
    """
    df = load_data(input_file)
    similarity_df = calculate_similarity(df, user_threshold)
    similarity_df.to_csv(output_file, index=False)

def load_data(input_file):
    """
    Load data into a DataFrame with necessary columns only.
    """
    df = pd.read_csv(input_file, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'], usecols=[0, 1, 2])
    return df

def calculate_similarity(df, user_threshold=5):
    """
    Calculate movie similarity using sparse matrix and cosine similarity.
    """
    # Creating a sparse matrix for movie-user ratings
    movie_user_matrix = df.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
    movie_user_sparse = csr_matrix(movie_user_matrix.values)

    # Calculate cosine similarity in a vectorized way
    similarity_matrix = cosine_similarity(movie_user_sparse)

    # Filter out low-common user pairs and convert results to DataFrame
    similarity_results = []
    for i, movie_id in enumerate(movie_user_matrix.index):
        similar_indices = np.where((similarity_matrix[i] > 0) & (np.count_nonzero(movie_user_sparse[i].toarray()) >= user_threshold))[0]
        
        for j in similar_indices:
            if i != j:  # Skip the same movie
                similarity_results.append((
                    movie_id, 
                    movie_user_matrix.index[j], 
                    similarity_matrix[i, j], 
                    np.count_nonzero(movie_user_sparse[i].multiply(movie_user_sparse[j]).toarray())
                ))

    # Convert results to DataFrame
    similarity_df = pd.DataFrame(similarity_results, columns=['base_movie_id', 'most_similar_movie_id', 'similarity_score', 'common_ratings_count'])
    return similarity_df

if __name__ == "__main__":
    input_file = "u.data"
    output_file = "similarity_output_3.csv"
    t1 = time.time()
    compute_similarity(input_file, output_file, user_threshold=5)
    print(f"Finished computing movie similarity in {time.time() - t1:.2f} seconds")
