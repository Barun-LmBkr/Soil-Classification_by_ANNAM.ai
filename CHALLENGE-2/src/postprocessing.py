# postprocessing.py
"""

Author: Annam.ai IIT Ropar
Team Name: Ice 'N' Dagger
Team Members: Barun Saha, Bibaswan Das
Leaderboard Rank: 70

"""

# Here you add all the post-processing related details for the task completed from Kaggle.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_submission(test_ids, predictions, output_path):
    """Create and save the submission file."""
    submission_df = pd.DataFrame({
        'image_id': test_ids,
        'label': predictions
    })
    submission_df.to_csv(output_path, index=False)
    print(f"✓ Submission file saved to: {output_path}")
    return submission_df


def analyze_confidence(test_ids, predictions, decision_scores, save_path=None):
    """Analyze confidence of predictions based on decision scores."""
    confidence_df = pd.DataFrame({
        'image_id': test_ids,
        'prediction': predictions,
        'decision_score': decision_scores
    })

    soil_scores = confidence_df[confidence_df['prediction'] == 1]['decision_score']
    non_soil_scores = confidence_df[confidence_df['prediction'] == 0]['decision_score']

    print("\nConfidence Analysis:")
    print(f"Soil (1): {len(soil_scores)} images | Mean: {soil_scores.mean():.4f}, Min: {soil_scores.min():.4f}, Max: {soil_scores.max():.4f}")
    print(f"Non-Soil (0): {len(non_soil_scores)} images | Mean: {non_soil_scores.mean():.4f}, Min: {non_soil_scores.min():.4f}, Max: {non_soil_scores.max():.4f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(soil_scores, bins=30, alpha=0.7, label='Soil (1)', color='brown')
    plt.hist(non_soil_scores, bins=30, alpha=0.7, label='Non-soil (0)', color='gray')
    plt.xlabel('Decision Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Decision Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.boxplot([soil_scores, non_soil_scores], labels=['Soil', 'Non-soil'])
    plt.ylabel('Decision Score')
    plt.title('Decision Score Box Plot')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")

    plt.show()
    return confidence_df
