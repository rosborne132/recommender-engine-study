import ssl
from surprise import Dataset, KNNBasic, accuracy, SVD
from surprise.model_selection import train_test_split

# Disable SSL certificate verification
# Needed to download the dataset
ssl._create_default_https_context = ssl._create_unverified_context

# Load custom dataset from within the surprise library
movie_data = Dataset.load_builtin('ml-100k')

# Split dataset into a 75%/25% train and test set
train_set, test_set = train_test_split(movie_data, test_size=.2, random_state=42)

# The ur method allows you to look inside the dataset with an index
# Format: ({item_id}, {rating})
train_set.ur[590]

# Train the recommender with the provided dataset
movie_recommender = KNNBasic()
movie_recommender.fit(train_set)

# Create an SVD recommender that uses the SVD algorithm
svd_recommender = SVD()
svd_recommender.fit(train_set)

# Evaluating the recommender performance
predictions = movie_recommender.test(test_set)
svd_predictions = svd_recommender.test(test_set)

# We can measure one aspect of the model’s performance by looking at
# the RMSE (root-mean square error). The RMSE is an average measure
# of how far off predictions will be from their actual values. The closer
# the RMSE is to 0, the more accurate the model.
accuracy.rmse(predictions)
accuracy.rmse(svd_predictions)