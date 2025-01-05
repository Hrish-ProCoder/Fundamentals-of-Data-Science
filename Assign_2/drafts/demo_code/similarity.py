import time
import pandas as pd
import numpy as np

def compute_similarity(input_file, output_file, user_threshold):
    """
    Function to compute similarity scores
    
    Arguments
    ---------
    input_file: str, path to input MovieLens file 
    output_file: str, path to output .csv 
    user_threshold: int, optional argument to specify
    the minimum number of common users between movies 
    to compute a similarity score. The default value 
    should be 5. 
    """

    ##################################################
    # YOUR CODE HERE
    df = load_data(input_file)
    similarity_df = calculate_similarity(df, user_threshold)
    similarity_df.to_csv(output_file, index=False)
    ##################################################

def load_data(input_file):
    ##################################################
    # YOUR CODE HERE
    df = pd.read_csv(input_file, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'], usecols=[0, 1, 2])
    return df
    ##################################################

def calculate_similarity(df, user_threshold=5):
    ##################################################
    # YOUR CODE HERE
    movie_user_matrix = df.pivot(index='movie_id', columns='user_id', values='rating')

    movie_means = movie_user_matrix.mean(axis=1)
    centered_ratings = movie_user_matrix.sub(movie_means, axis = 0)

    similarity_results = []

    for i, movie_a in enumerate(centered_ratings.index):
        max_similarity = -2
        most_similar_movie = None
        common_count = 0
        
        for movie_b in centered_ratings.index[i+1:]:

            common_users = centered_ratings.loc[[movie_a, movie_b]].dropna(axis=1)
            if common_users.shape[1] < user_threshold:
                continue
            
            # Calculate adjusted cosine similarity
            ratings_a = common_users.loc[movie_a]
            ratings_b = common_users.loc[movie_b]
            numerator = np.sum(ratings_a * ratings_b)
            denominator = np.sqrt(np.sum(ratings_a ** 2)) * np.sqrt(np.sum(ratings_b ** 2))
            
            if denominator != 0:
                similarity = numerator / denominator
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_movie = movie_b
                    common_count = common_users.shape[1]
        
        # Store result
        similarity_results.append((movie_a, most_similar_movie, max_similarity, common_count))
    
    # Convert results to DataFrame
    similarity_df = pd.DataFrame(similarity_results, columns=['base_movie_id', 'most_similar_movie_id', 'similarity_score', 'common_ratings_count'])
    return similarity_df

    ##################################################



if __name__ == "__main__":
    input_file = "u.data"
    output_file = "similarity_output.csv"
    t1 = time.time()
    compute_similarity(input_file, output_file, user_threshold=5)
    print(f"Finished computing movie similarity in {time.time() - t1:.2f} seconds")