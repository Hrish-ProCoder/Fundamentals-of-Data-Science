import time
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(input_file, output_file, user_threshold=5):
    """
    Function to compute similarity scores
    
    Arguments
    ---------
    input_file: str, path to input MovieLens file 
    output_file: str, path to output .csv 
    user_threshold: int, optional argument to specify
    the minimum number of common users between movies 
    to compute a similarity score. Default is 5.
    """
    
    # Load data
    df = pd.read_csv(input_file, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'], usecols=[0, 1, 2])
    
    # Construct the movie-user matrix
    movie_user_matrix = df.pivot(index='movie_id', columns='user_id', values='rating')
    movie_user_matrix = movie_user_matrix.fillna(0)
    
    # Center the ratings by subtracting mean rating per movie
    movie_user_matrix_centered = movie_user_matrix.sub(movie_user_matrix.mean(axis=1), axis=0)
    
    # Convert to sparse matrix format for efficient similarity computation
    sparse_matrix = csr_matrix(movie_user_matrix_centered.values)
    
    # Compute cosine similarity between movies
    similarity_matrix = cosine_similarity(sparse_matrix)
    np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
    
    # Retrieve pairs with similarity scores and common user counts
    similarity_results = []
    movie_ids = movie_user_matrix.index.values
    
    for i in range(len(movie_ids)):
        for j in range(i + 1, len(movie_ids)):
            common_users = (movie_user_matrix.iloc[i] != 0) & (movie_user_matrix.iloc[j] != 0)
            common_count = common_users.sum()
            
            if common_count >= user_threshold:
                similarity_score = similarity_matrix[i, j]
                similarity_results.append((movie_ids[i], movie_ids[j], similarity_score, common_count))
    
    # Convert results to DataFrame and save to CSV
    similarity_df = pd.DataFrame(similarity_results, columns=['base_movie_id', 'most_similar_movie_id', 'similarity_score', 'common_ratings_count'])
    similarity_df.to_csv(output_file, index=False)

# Usage example
if __name__ == "__main__":
    input_file = "u.data"
    output_file = "similarity_output_2.csv"
    t1 = time.time()
    compute_similarity(input_file, output_file, user_threshold=5)
    print(f"Finished computing movie similarity in {time.time() - t1:.2f} seconds")
