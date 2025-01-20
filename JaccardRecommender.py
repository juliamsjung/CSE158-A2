# %%
import gzip
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = []
file_path = "renttherunway_final_data.json.gz"

with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding line: {e}")

df = pd.DataFrame(data)
df.head()

# %%
df.isnull().sum().sum()

# %% [markdown]
# ## Data Cleaning

# %%
### Data Overview
overview = df.info()
overview

# %%
### Descriptive Statistics
descriptive_stats = df.describe(include="all")
descriptive_stats

# %% [markdown]
# ## Convert Height to Float

# %%
df = pd.DataFrame(data)

df['height'].isna().sum()
# small null values for height; we can just drop
df = df.dropna(subset=['height'])
df['height'].isna().sum()

def height_to_float(height):
    feet, inches = height.split("' ")
    inches = inches.replace("\"", "")
    return float(feet) * 12 + float(inches)

# small null values for weight; we can just drop
df['weight'].isna().sum()
df['height (in)'] = df['height'].apply(height_to_float)

def clean_weight(value):
    if pd.isna(value):
        return np.nan
    return float(value.replace('lbs', ''))
df['weight'] = df['weight'].apply(clean_weight)

mean_weights_by_height = df.groupby('height (in)')['weight'].mean()
df['weight'] = df.apply(
    lambda row: mean_weights_by_height[row['height (in)']] if pd.isna(row['weight']) else row['weight'],
    axis=1
)
df = df.dropna(subset=['rating', 'rented for', 'body type', 'age'])

df['age'] = df['age'].astype(int)

df = df[['fit', 'user_id', 'item_id', 'weight', 'rating', 'rented for', 'body type', 'category', 'height (in)', 'size', 'age']]

# %% [markdown]
# ## Data Visualization

# %%
fit_counts = df["fit"].value_counts()
plt.figure(figsize=(8, 8))
fit_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
plt.title("Distribution of 'fit'")
plt.ylabel("")
plt.show()

# %%
categorical_columns = ["rented for", "body type"]

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
for i, col in enumerate(categorical_columns):
    data = df[col].value_counts()
    axes[i].pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    axes[i].set_title(f"Distribution of {col}")
plt.tight_layout()


# %%
categorical_columns = ["rented for", "body type"]

for col in categorical_columns:
    data = df[col].value_counts()
    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(
        data, 
        labels=data.index, 
        autopct=lambda pct: f'{pct:.1f}%',  # Ensures percentage is always calculated
        startangle=90, 
        colors=sns.color_palette("pastel"),
        textprops={'fontsize': 12, 'color': 'black'}
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('black')  # Ensures visibility of text inside slices
    plt.title(f"Distribution of {col}")
    plt.show()


# %%
numeric_columns = ["rating", "size", "age"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(numeric_columns):
    sns.histplot(df[col], bins=25, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
plt.tight_layout()

# %%
df['rating'] = df['rating'].astype(int)
rating_counts = df['rating'].value_counts().sort_index()
rating_counts

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Create a figure and axis for the plot
plt.figure(figsize=(8, 6))

# Plot the histogram for the 'rating' column
ax = sns.histplot(df['rating'], bins=4, kde=False)

# Add title and labels
plt.xlabel("Rating")
plt.ylabel("Frequency")

# Get the counts and edges of the bins
counts, bins = np.histogram(df['rating'], bins=4)

# Add count labels above each bar
for i in range(len(counts)):
    ax.text(bins[i] + 0.5, counts[i] + 1, str(counts[i]), ha='center', va='bottom')

# Display the plot
plt.tight_layout()
plt.show()


# %%
import pandas as pd
from collections import defaultdict

# Constructing mappings for users and items
usersPerItem = defaultdict(set)
itemsPerUser = defaultdict(set)

for _, row in df.iterrows():
    user, item = row['user_id'], row['item_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)

# Recommender logic with progress updates
def jaccard_similarity(item1, item2):
    """Compute Jaccard similarity between two items."""
    users1 = usersPerItem[item1]
    users2 = usersPerItem[item2]
    numer = len(users1.intersection(users2))
    denom = len(users1.union(users2))
    return numer / denom if denom != 0 else 0

def compute_score(user, item):
    """Compute a recommendation score for a given user and item."""
    max_similarity = 0
    for existing_item in itemsPerUser[user]:
        similarity = jaccard_similarity(item, existing_item)
        max_similarity = max(max_similarity, similarity)
    
    # Popularity score as the number of users who interacted with the item
    popularity_score = len(usersPerItem[item])
    
    # Weighted score
    total_score = 5 * max_similarity + popularity_score
    return total_score

# Generate predictions with progress updates
user_item_scores = defaultdict(list)
processed_count = 0
progress_step = 50000  # Print progress every 50,000 iterations

for _, row in df.iterrows():
    user, item = row['user_id'], row['item_id']
    score = compute_score(user, item)
    user_item_scores[user].append((item, score))

    # Progress update
    processed_count += 1
    if processed_count % progress_step == 0:
        print(f"Processed {processed_count} rows.")

# Sorting predictions for each user
predictions = {}
for user, items in user_item_scores.items():
    items.sort(key=lambda x: x[1], reverse=True)
    num_items = len(items)
    num_read = num_items // 2
    
    for i, (item, _) in enumerate(items):
        predictions[(user, item)] = 1 if i < num_read else 0

# Displaying predictions
for (user, item), prediction in predictions.items():
    print(f"User {user} -> Item {item}: {'Recommended' if prediction == 1 else 'Not Recommended'}")


# %%
temp = pd.DataFrame(
    list(predictions.items()),
    columns=['Pair', 'Value']
)

# Split the tuple into separate columns
temp[['Key1', 'Key2']] = pd.DataFrame(df['Pair'].tolist(), index=df.index)

# Drop the original 'Pair' column
temp = temp.drop(columns=['Pair'])

# Rearrange columns if needed
temp = temp[['Key1', 'Key2', 'Value']]

temp

# %%
# Convert user_item_scores to a DataFrame
recommendations = user_item_scores

# Create DataFrame
sorted_scores = pd.DataFrame(predictions)

# Sort by user_id and similarity_score in descending order
sorted_scores = sorted_scores.sort_values(by=['user_id', 'similarity_score'], ascending=[True, False])


def calculate_accuracy(df, k):
    """
    Calculate the accuracy for top-K predictions when all user-item pairs in the dataset are ground truth.
    Args:
        df: DataFrame containing user_id, item_id, and similarity_score.
        k: Number of top recommendations to consider.
    Returns:
        Accuracy as a fraction of correctly predicted recommendations.
    """
    correct_predictions = 0
    total_predictions = 0

    for user, group in df.groupby('user_id'):
        # Top K recommendations for the user
        top_k = group.nlargest(k, 'similarity_score')
        recommended_items = set(top_k['item_id'])
        
        # Ground truth for this user (all items associated with the user in the dataset)
        truth_items = set(group['item_id'])
        
        # Count correct predictions
        correct_predictions += len(recommended_items.intersection(truth_items))
        total_predictions += k

    # Accuracy calculation
    return correct_predictions / total_predictions if total_predictions > 0 else 0

# Example usage with k=10
k = 5
accuracy = calculate_accuracy(sorted_scores, k)
print(f"Accuracy@{k}: {accuracy:.4f}")


# %%
# l = 5, p = 0.3251

# %%
def calculate_precision_recall(df, k):
    """
    Calculate precision and recall for top-K predictions.
    Args:
        df: DataFrame containing user_id, item_id, and similarity_score.
        k: Number of top recommendations to consider.
    Returns:
        Tuple containing average precision and recall across all users.
    """
    total_precision = 0
    total_recall = 0
    user_count = 0

    for user, group in df.groupby('user_id'):
        # Top K recommendations for the user
        top_k = group.nlargest(k, 'similarity_score')
        recommended_items = set(top_k['item_id'])
        
        # Ground truth for this user (all items associated with the user in the dataset)
        truth_items = set(group['item_id'])
        
        # Count correct predictions
        correct_predictions = len(recommended_items.intersection(truth_items))
        
        # Precision and Recall for this user
        precision = correct_predictions / k if k > 0 else 0
        recall = correct_predictions / len(truth_items) if len(truth_items) > 0 else 0
        
        total_precision += precision
        total_recall += recall
        user_count += 1

    # Average precision and recall across all users
    avg_precision = total_precision / user_count if user_count > 0 else 0
    avg_recall = total_recall / user_count if user_count > 0 else 0

    return avg_precision, avg_recall

# Example usage with k=5
k = 5
precision, recall = calculate_precision_recall(sorted_scores, k)
print(f"Precision@{k}: {precision:.4f}")
print(f"Recall@{k}: {recall:.4f}")


# %%
import pandas as pd
import random

# Function to generate random recommendations
def random_recommender(df, k):
    """
    Generate random recommendations for each user.
    
    Args:
        df: DataFrame containing user_id, item_id.
        k: Number of recommendations to generate for each user.
    
    Returns:
        DataFrame with random recommendations.
    """
    recommendations = []

    # For each unique user in the dataset
    for user in df['user_id'].unique():
        # Get all items
        all_items = df['item_id'].unique()
        
        # Randomly sample 'k' items for the user (with replacement allowed)
        random_items = random.sample(list(all_items), k)
        
        # Add recommendations to the list
        for item in random_items:
            recommendations.append((user, item))

    # Create DataFrame from the random recommendations
    recommendations_df = pd.DataFrame(recommendations, columns=['user_id', 'item_id'])
    return recommendations_df

# Generate random recommendations for each user
k = 3  # Number of random recommendations per user
random_recs = random_recommender(df, k)

# Show the random recommendations
print(random_recs)


# %%
from sklearn.metrics import precision_score, recall_score, f1_score

# Assuming df contains the actual interactions (user_id, item_id)
# and random_recs contains the recommended items

# Merge the actual interactions with the recommendations
merged = pd.merge(df, random_recs, on=['user_id', 'item_id'], how='right', indicator=True)

# Calculate true positives, false positives, and false negatives
merged['actual_interaction'] = merged['_merge'] == 'both'

# Precision, recall, F1 score calculation
# True positives are when 'actual_interaction' is True, and the recommendation exists
precision = precision_score(merged['actual_interaction'], merged['item_id'].notna(), average='binary')
recall = recall_score(merged['actual_interaction'], merged['item_id'].notna(), average='binary')
f1 = f1_score(merged['actual_interaction'], merged['item_id'].notna(), average='binary')

# Print out the metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# %%
import pandas as pd

# Function to generate popularity-based recommendations
def popularity_recommender(df, k):
    """
    Generate popularity-based recommendations for each user.
    
    Args:
        df: DataFrame containing user_id, item_id.
        k: Number of recommendations to generate for each user.
    
    Returns:
        DataFrame with popularity-based recommendations.
    """
    recommendations = []

    # Calculate the popularity of each item based on interactions
    item_popularity = df['item_id'].value_counts().reset_index()
    item_popularity.columns = ['item_id', 'popularity']
    
    # For each unique user in the dataset
    for user in df['user_id'].unique():
        # Get the items the user has already interacted with
        user_items = df[df['user_id'] == user]['item_id'].unique()
        
        # Get the top 'k' popular items that the user has not interacted with
        popular_items = item_popularity[~item_popularity['item_id'].isin(user_items)].head(k)
        
        # Add recommendations to the list
        for _, row in popular_items.iterrows():
            recommendations.append((user, row['item_id']))

    # Create DataFrame from the popularity-based recommendations
    recommendations_df = pd.DataFrame(recommendations, columns=['user_id', 'item_id'])
    return recommendations_df

# Generate popularity-based recommendations for each user
k = 3  # Number of recommendations per user
popularity_recs = popularity_recommender(df, k)

# Show the popularity-based recommendations
print(popularity_recs)


# %%
import pandas as pd

# Function to calculate precision and recall
def calculate_precision_recall(df, recommendations_df, k):
    """
    Calculate precision and recall for popularity-based recommendations.
    
    Args:
        df: Original DataFrame containing user_id and item_id (ground truth).
        recommendations_df: DataFrame with recommendations.
        k: Number of recommendations per user.
    
    Returns:
        Precision and Recall values.
    """
    total_relevant = 0  # Count of relevant (recommended and interacted) items
    total_recommended = 0  # Count of all recommended items
    total_interacted = 0  # Count of all items users have interacted with
    
    for user in df['user_id'].unique():
        # Ground truth: items the user has interacted with
        ground_truth_items = set(df[df['user_id'] == user]['item_id'])
        total_interacted += len(ground_truth_items)
        
        # Recommended items for this user
        recommended_items = set(recommendations_df[recommendations_df['user_id'] == user]['item_id'])
        total_recommended += len(recommended_items)
        
        # Relevant recommendations
        relevant_items = recommended_items.intersection(ground_truth_items)
        total_relevant += len(relevant_items)
    
    # Precision and Recall calculations
    precision = total_relevant / total_recommended if total_recommended > 0 else 0
    recall = total_relevant / total_interacted if total_interacted > 0 else 0
    
    return precision, recall


# Generate popularity-based recommendations
k = 3  # Number of recommendations per user
popularity_recs = popularity_recommender(df, k)

# Calculate precision and recall
precision, recall = calculate_precision_recall(df, popularity_recs, k)

# Display results
print(f"Precision@{k}: {precision:.4f}")
print(f"Recall@{k}: {recall:.4f}")


# %%
from sklearn.metrics import precision_score, recall_score, f1_score

# True positives: 'actual_interaction' is True, and the recommendation exists
precision = precision_score(merged['actual_interaction'], merged['item_id'].notna(), average='binary')
recall = recall_score(merged['actual_interaction'], merged['item_id'].notna(), average='binary')
f1 = f1_score(merged['actual_interaction'], merged['item_id'].notna(), average='binary')

# Print out the metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# %%
df.describe()

# %%



