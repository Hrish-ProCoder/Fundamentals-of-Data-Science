

 1. What were your considerations when creating this test data?

**Ans. 1:**

+ Rating diversity and Variety of user-movie interactions which is observed in real data (u.data) as well.

+ Included a mix of high and low ratings.

+ Adding some repetitive ratings can also challenge the written code as I have handled particular cases.

+ Ensured multiple users rated the same movies for meaningful similarity calculations.





 2. Were there certain characteristics of the real data and file format that you made sure to capture in your test data?

**Ans. 2:**

+ I have replicated structure with columns: user ID, movie ID, rating, timestamp.

+ Maintained realistic user-movie pair for similarity checks.

+ I have used same tab-delimited format as in the real data.

+ Included common ratings across movies to facitilate similarity function testing.





 3. Did you create a reference solution for your test data? If so, how?

**Ans. 3:**

+ I manually prepared the dataset such a way that it matches the expected similarity scores.

+ I have Debugged the code to get a preview of movie comparisons and related common users. (Commented in code)

 print(f"Comparing Movie {movie_ids[i]} and Movie {movie_ids[j]}: Common Users = {common_count}")

+ Verified common ratings counts for selected pairs.

+ Added time taken or runtime to get movie similarity (in secs).

+ Done Manual testing on the test dataset which helped me to verify with the results.

**My Output for the test data is:**

base_movie_id,most_similar_movie_id,similarity_score,common_ratings_count
1,3,1.0,2
3,2,-0.7217976369344361,4
