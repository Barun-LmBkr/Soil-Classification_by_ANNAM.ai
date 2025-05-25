# postprocessing.py
"""

Author: Annam.ai IIT Ropar
Team Name: Ice 'N' Dagger
Team Members: Barun Saha, Bibaswan Das
Leaderboard Rank: 70

"""

# Here you add all the post-processing related details for the task completed from Kaggle.


import pandas as pd
import joblib

# Optional file if additional postprocessing is needed

# Example: Load predictions and re-encode labels (already done in inference normally)
submission = pd.read_csv("submission.csv")
print(submission.head())  # Verify content

# You could save results in a different format or analyze them here
