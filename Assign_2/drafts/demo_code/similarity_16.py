import time
import pandas as pd
import numpy as np

def load_data(input_file):

    """
    Function to load and preprocess data
    
    Arguments
    ---------
    input_file: str, path to input MovieLens file which is  u.data

    Returns : Centered Ratings and mean of the movies which will be further used
    """ 

    ratings_data = pd.read_csv(
        input_file, 
        sep='\t', 
        names = ['user_id', 'movie_id', 'rating', 'timestamp'], 
        usecols = [0, 1, 2]
    )
    user_ids = ratings_data['user_id'].unique()
    movie_ids = ratings_data['movie_id'].unique()
    
    # Mapping both ids to index - using enumerate
    user_index_id = {user_id: ids for ids, user_id in enumerate(user_ids)}
    movie_index_id = {movie_id: ids for ids, movie_id in enumerate(movie_ids)}
    
    num_movies = len(movie_ids)
    num_users = len(user_ids)
    ratings_matrix = np.zeros((num_movies, num_users), dtype=np.float32)  # Relevant Matrix with Zeros will be generated
    
    for row in ratings_data.itertuples():                     # Iterate over each row - used intertuples
        movie_idx = movie_index_id[row.movie_id]             # Filling the rating_matrix with particular ratings
        user_idx = user_index_id[row.user_id]                
        ratings_matrix[movie_idx, user_idx] = row.rating   
    
    return ratings_matrix, user_ids, movie_ids, user_index_id, movie_index_id

def center_ratings(ratings_matrix):

    """
    Function to find centered rating w.r.t users and movies linkage
    
    Arguments
    ---------
    ratings_matrix: All the rating matrix are filled with particular ratings

    Returns : Centered Ratings and Mean of the Movies which will be further used
    """ 

    movie_means = np.true_divide(ratings_matrix.sum(axis=1), (ratings_matrix != 0).sum(axis=1))
    movie_means[np.isnan(movie_means)] = 0  # This will handle movies with no ratings which is not the case for u.data as it is already clean
    
    centered_ratings = ratings_matrix - movie_means[:, np.newaxis]
    centered_ratings[ratings_matrix == 0] = 0                       # Manages absent/missing values
    
    return centered_ratings, movie_means

def calculate_similarity(i, j, centered_ratings, user_threshold):

    """
    Function to calculate similarity and common count between movies
    
    Arguments
    ---------
    i, j: Retrieves movies i and j and identify common users and will help iterating in centered_ratings
    centered_ratings: Centers each movie by subtracting movie mean
    user_threshold: int, optional argument to specify
    the minimum number of common users between movies 
    to compute a similarity score.
    
    Considered user threshold value as 5. 

    Returns : Similarity and Common count for movies
    """    
    
    ratings_i, ratings_j = centered_ratings[i], centered_ratings[j]
    common_users = (ratings_i != 0) & (ratings_j != 0)
    common_count = np.count_nonzero(common_users)       # Identifying common users who have rated both the movies
    
    if common_count < user_threshold:    # Considered 5
        return None, common_count
    
    ratings_i_common = ratings_i[common_users]          # Rating Checking for common useres for both the movies 
    ratings_j_common = ratings_j[common_users]
    
    numerator = np.sum(ratings_i_common * ratings_j_common)
    denominator = np.sqrt(np.sum(ratings_i_common ** 2) * np.sum(ratings_j_common ** 2))
    
    if denominator != 0:                        
        similarity = numerator / denominator
    else:
        similarity = 0          
    
    return similarity, common_count

def compute_similarity(input_file, output_file, user_threshold=5):

    """
    Function to compute similarity scores
    
    Arguments
    ---------
    input_file: str, path to input MovieLens file which is u.data
    output_file: str, path to output .csv 
    user_threshold: int, optional argument to specify
    the minimum number of common users between movies 
    to compute a similarity score.
    Considered user threshold value as 5. 
    """

    ratings_matrix, user_ids, movie_ids, user_index_id, movie_index_id = load_data(input_file)
    num_movies = len(movie_ids)
    
    centered_ratings, movie_means = center_ratings(ratings_matrix)
    
    similarity_results = []                           # Store similarity results
    
    for i in range(num_movies):                       # Iterate over i,j to check if enough rating is shared between them considering threshold value
        for j in range(i + 1, num_movies):
            similarity, common_count = calculate_similarity(i, j, centered_ratings, user_threshold)
            # print(f"Comparing Movie {movie_ids[i]} and Movie {movie_ids[j]}: Common Users = {common_count}")  # Used for Debugging 
            
            if similarity is not None and movie_ids[i] <= 20:                                          # Considering base_movie_id <= 20 , using this instead of .head()
                similarity_results.append((movie_ids[i], movie_ids[j], similarity, common_count))
    
    similarity_data_info = pd.DataFrame(similarity_results, columns=['base_movie_id', 'most_similar_movie_id', 'similarity_score', 'common_ratings_count'])
 
  
    # Filtering to keep only the top 1 movie for each base_movie_id ------ based on highest similarity_score
    top_similarities = similarity_data_info[similarity_data_info['base_movie_id'] <= 20]
    top_similarities = top_similarities.loc[top_similarities.groupby('base_movie_id')['similarity_score'].idxmax()]

    top_similarities.to_csv(output_file, index=False)
    print("CSV was successfully downloaded")

    print("Average common ratings count:", top_similarities['common_ratings_count'].mean())    # For Reference while code testing







if __name__ == "__main__":
    input_file = "u.data"
    output_file = "similarity_output_1.csv"
    t1 = time.time()
    compute_similarity(input_file, output_file, user_threshold=5)
    print(f"Finished computing movie similarity in {time.time() - t1:.2f} seconds")
