import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample Data: Movie Ratings
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4],
    'movie_id': [101, 102, 103, 101, 104, 102, 103, 104, 101, 102, 104],
    'rating': [5, 4, 3, 5, 4, 4, 3, 5, 2, 3, 4]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create a pivot table (user-item matrix)
pivot_table = df.pivot_table(index='user_id', columns='movie_id', values='rating')

# Fill NaN with 0 (for simplicity)
pivot_table.fillna(0, inplace=True)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(pivot_table)

# Convert to DataFrame for better readability
user_similarity_df = pd.DataFrame(user_similarity, index=pivot_table.index, columns=pivot_table.index)

# Function to make predictions
def predict_ratings(user_id, movie_id):
    similar_users = user_similarity_df[user_id]
    ratings = pivot_table[movie_id]
    
    # Weighted sum of ratings of similar users
    weighted_sum = np.dot(similar_users, ratings)
    
    # Sum of similarities
    sum_of_similarities = np.sum(similar_users)
    
    # Prediction
    if sum_of_similarities == 0:
        return 0
    else:
        return weighted_sum / sum_of_similarities

# Example: Predict the rating for user 1 and movie 104
predicted_rating = predict_ratings(user_id=1, movie_id=104)
print(f"Predicted Rating for User 1 and Movie 104: {predicted_rating:.2f}")
