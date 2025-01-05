import time
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def compute_similarity(input_file, output_file, user_threshold=5):
    """
    Function to compute similarity scores
    
    Arguments:
    - input_file: str, path to input MovieLens file 
    - output_file: str, path to output .csv 
    - user_threshold: int, minimum number of common users between movies 
                      to compute a similarity score. Default is 5.
    """
    
    # Load and preprocess data
    df = pd.read_csv(input_file, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'], usecols=[0, 1, 2])
    
    # Create mappings of user and movie indices
    user_ids = df['user_id'].unique()
    movie_ids = df['movie_id'].unique()
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    
    # Initialize the movie-user ratings matrix
    num_movies = len(movie_ids)
    num_users = len(user_ids)
    ratings_matrix = np.zeros((num_movies, num_users), dtype=np.float32)
    
    # Fill the ratings matrix
    for row in df.itertuples():
        movie_idx = movie_to_idx[row.movie_id]
        user_idx = user_to_idx[row.user_id]
        ratings_matrix[movie_idx, user_idx] = row.rating
    
    # Center the ratings by subtracting mean rating per movie
    movie_means = np.true_divide(ratings_matrix.sum(axis=1), (ratings_matrix != 0).sum(axis=1))
    movie_means[np.isnan(movie_means)] = 0  # Handle movies with no ratings
    centered_ratings = ratings_matrix - movie_means[:, np.newaxis]
    centered_ratings[ratings_matrix == 0] = 0  # Zero out entries with no ratings
    
    # Compute similarity matrix
    similarity_results = []
    for i in range(num_movies):
        ratings_i = centered_ratings[i]
        
        for j in range(i + 1, num_movies):
            ratings_j = centered_ratings[j]
            
            # Identify common users
            common_users = (ratings_i != 0) & (ratings_j != 0)
            common_count = np.count_nonzero(common_users)
            
            # Skip if common users are below the threshold
            if common_count < user_threshold:
                continue
            
            # Calculate adjusted cosine similarity
            ratings_i_common = ratings_i[common_users]
            ratings_j_common = ratings_j[common_users]
            
            numerator = np.sum(ratings_i_common * ratings_j_common)
            denominator = np.sqrt(np.sum(ratings_i_common ** 2) * np.sum(ratings_j_common ** 2))
            
            # Handle cases where denominator is zero
            similarity = numerator / denominator if denominator != 0 else 0
            
            # Append results for base_movie_id from 1 to 20
            if movie_ids[i] <= 20:
                similarity_results.append((movie_ids[i], movie_ids[j], similarity, common_count))
    
    # Convert results to DataFrame
    similarity_df = pd.DataFrame(similarity_results, columns=['base_movie_id', 'most_similar_movie_id', 'similarity_score', 'common_ratings_count'])
    
    # Filter to keep only the top 1 movie for each base_movie_id based on highest similarity_score
    top_similarities = similarity_df[similarity_df['base_movie_id'] <= 20]
    top_similarities = top_similarities.loc[top_similarities.groupby('base_movie_id')['similarity_score'].idxmax()]

    # Save to CSV
    top_similarities.to_csv(output_file, index=False)

    print("Average common ratings count:", top_similarities['common_ratings_count'].mean())

# Run the program
if __name__ == "__main__":
    input_file = "u.data"  # Update with your file path
    output_file = "similarity_output11.csv"
    t1 = time.time()
    compute_similarity(input_file, output_file, user_threshold=5)
    print(f"Finished computing movie similarity in {time.time() - t1:.2f} seconds")
