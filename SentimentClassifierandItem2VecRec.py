# %%
import nltk
nltk.download('stopwords')

# %%
import gzip
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string


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
df['review_summary'].isna().sum()

# %%
df['review_text'].isna().sum()

# %%
df = df.dropna(subset=['rating'])

# %%
df['rating'].isna().sum()

# %% [markdown]
# ## Data Cleaning

# %%
overview = df.info()
overview

# %%
### Descriptive Statistics
descriptive_stats = df.describe(include="all")
descriptive_stats

# %% [markdown]
# ### Convert Height to Float and Remove 'lbs' from 'weight' Column

# %%
df['height'].isna().sum()

# %%
# small null values for height; we can just drop
df = df.dropna(subset=['height'])

# %%
df['height'].isna().sum()

# %%
def height_to_float(height):
    feet, inches = height.split("' ")
    inches = inches.replace("\"", "")
    return float(feet) + float(inches) / 12

# %%
df['weight']

# %%
# too many null values for weight, cant drop them
df['weight'].isna().sum()

# %%
def clean_weight(value):
    if pd.isna(value):
        return np.nan
    return float(value.replace('lbs', ''))
df['weight'] = df['weight'].apply(clean_weight)

# %%
df['weight']

# %%
df['height (in)'] = df['height'].apply(height_to_float)

# %%
df['height (in)']

# %% [markdown]
# ### Replace missing 'weight' values with the mean of the weight for a given height

# %%
mean_weights_by_height = df.groupby('height (in)')['weight'].mean()

# %%
df['weight'] = df.apply(
    lambda row: mean_weights_by_height[row['height (in)']] if pd.isna(row['weight']) else row['weight'],
    axis=1
)

# %%
df['weight'].isna().sum()

# %% [markdown]
# ### Dropping Nulls for Columns that Don't Have a lot of Nulls

# %%
# not a lot of null values for age, we can drop
df['age'].isna().sum()

# %%
df = df.dropna(subset=['age'])

# %%
df.head()

# %%
df = df.dropna(subset=['rating'])

# %%
df['rating'].isna().sum()

# %%
df.shape

# %% [markdown]
# ### Text Preprocessing (removing stopwords, punctuation, converting to Word2Vec Dense Embeddings)

# %%
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in punctuation])
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return tokens
df['cleaned_text'] = df['review_text'].apply(clean_text)

# %%
df['cleaned_text'].iloc[1]

# %%
df['review_text'].iloc[1]

# %%
from gensim.models import Word2Vec

model = Word2Vec(sentences=df['cleaned_text'], vector_size=100, window=5, min_count=5, sg=1)

def get_review_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

df['review_embedding'] = df['cleaned_text'].apply(lambda x: get_review_vector(x, model))

# %% [markdown]
# ### Encode Classes (Rating)

# %%
df['rating'].unique()

# %%
rating_classes = df['rating'].unique()
encoding_dict = {}

for i in range(len(rating_classes)):
    cur_class = rating_classes[i]
    encoding_dict[cur_class] = i

# %%
encoding_dict

# %%
df['rating_label'] = df['rating'].map(encoding_dict)

# %% [markdown]
# ### Split Train/Test Data (80/20)

# %%
X = np.vstack(df['review_embedding'].values)
y = df['rating_label']

# %%
len(X)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %% [markdown]
# ### Model 1 to Predict Ratings: Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)


# %%
inverse_encoding_dict = {v: k for k, v in encoding_dict.items()}
target_names = [inverse_encoding_dict[i] for i in range(len(inverse_encoding_dict))]
print(target_names)


# %%
from sklearn.metrics import classification_report, accuracy_score

y_pred = dt.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))

# %% [markdown]
# ### Decision Tree Hyperparameter Tuning

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'criterion': ['gini', 'entropy']
}


dt = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)


# %%
best_params = grid_search.best_params_

best_dt = DecisionTreeClassifier(**best_params, random_state=42)
best_dt.fit(X_train, y_train)

y_pred = best_dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))


# %%
len(X[0])

# %% [markdown]
# ### Model 2: Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=2,
    random_state=42,
    criterion='gini'
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))


# %% [markdown]
# ### Using 'review_summary' as an additional feature (convert to embedding and add to embedding data)

# %%
# Clean and tokenize the `review_summary` column
def preprocess_text(text):
    punctuation = set(string.punctuation)
    text = ''.join([c for c in text.lower() if c not in punctuation])
    return text.split()

df['cleaned_summary'] = df['review_summary'].apply(preprocess_text)

# %%
# Train Word2Vec on `cleaned_summary`
summary_model = Word2Vec(sentences=df['cleaned_summary'], vector_size=100, window=5, min_count=5, sg=1)

# %%
# Create embeddings for `review_summary`
def get_summary_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

df['summary_embedding'] = df['cleaned_summary'].apply(lambda x: get_summary_vector(x, summary_model))


# %%
# stack embeddings into a single feature matrix
review_embeddings = np.vstack(df['review_embedding'].values)  # embeddings for review_text
summary_embeddings = np.vstack(df['summary_embedding'].values)  # embeddings for review_summary

# combine both review and summary embeddings
X2 = np.hstack((review_embeddings, summary_embeddings))


# %% [markdown]
# ### Train/Test Split (80/20) using new embeddings data (both review and summary embeddings as one embedding)

# %%
from sklearn.model_selection import train_test_split

y = df['rating_label']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)


# %% [markdown]
# ### Model 1: Decision Tree (with new data)

# %%
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %% [markdown]
# ### Decision Tree Hyperparameter Tuning (on new data)

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'criterion': ['gini', 'entropy']
}


dt = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)


# %%
X_train[0]

# %%
best_params = grid_search.best_params_

# train new model with the best parameters
best_dt = DecisionTreeClassifier(**best_params, random_state=42)
best_dt.fit(X_train, y_train)

# evaluating
y_pred = best_dt.predict(X_test)

print("Best Parameters for Decision Tree:", grid_search.best_params_)
print("Accuracy for Tuned Decision Tree:", accuracy_score(y_test, y_pred))
print("Evaluation Metrics for Tuned Decision Tree:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))


# %%
best_dt

# %%
y_train_pred = best_dt.predict(X_train)

print(f"Fine-Tuned Decision Tree Accuracy on Training Set: {accuracy_score(y_train, y_train_pred)}")
print("Fine-Tuned Decision Tree Evaluation Metrics on Training Set:")
print(classification_report(y_train, y_train_pred, target_names=target_names, zero_division=0))


# %% [markdown]
# ### Visualizing Accuracy for Different Parameter Combinations (Decision Tree)

# %%
import matplotlib.pyplot as plt
import numpy as np

results = grid_search.cv_results_

param_combinations = results['params']
mean_train_scores = results['mean_train_score'] if 'mean_train_score' in results else results['mean_test_score']  # Some versions don't calculate mean_train_score
mean_test_scores = results['mean_test_score']
std_test_scores = results['std_test_score']

x_labels = [
    f"depth={params['max_depth']}, split={params['min_samples_split']}, leaf={params['min_samples_leaf']}, crit={params['criterion']}"
    for params in param_combinations
]

sorted_indices = np.argsort(mean_test_scores)
x_labels = np.array(x_labels)[sorted_indices]
mean_train_scores = np.array(mean_train_scores)[sorted_indices]
mean_test_scores = np.array(mean_test_scores)[sorted_indices]

plt.figure(figsize=(15, 7))
x = np.arange(len(x_labels))

plt.plot(x, mean_test_scores, label="Validation Accuracy", marker='o')

plt.xticks(x, x_labels, rotation=90)
plt.xlabel("Parameter Combinations")
plt.ylabel("Accuracy")
plt.title("Accuracy for Different Parameter Combinations (Decision Tree)")
plt.legend()
plt.tight_layout()

plt.show()


# %% [markdown]
# ### Model 2: Random Forest (on new data)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=2,
    random_state=42,
    criterion='gini'
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))


# %% [markdown]
# ### Random Forest Hyperparameter Tuning (again, on new data)

# %%
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [8, 10, 12],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42, criterion='gini')

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)


# %% [markdown]
# ### Random Foret Hyperparameter Tuning, Refined on a Specific Range for each Hyperparameter
# Decided to use these ranges as they seem to garner better performance during tuning.

# %%
param_grid_refined = {
    'n_estimators': [200, 250, 300],
    'max_depth': [20, 25],
    'min_samples_split': [2],
    'criterion': ['gini', 'entropy']
}

rf = RandomForestClassifier(random_state=42)

grid_search_refined = GridSearchCV(estimator=rf, param_grid=param_grid_refined, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_refined.fit(X_train, y_train)

print("Refined Best Parameters:", grid_search_refined.best_params_)
print("Refined Best Accuracy:", grid_search_refined.best_score_)


# %% [markdown]
# ### Evaluating Performance on Tuned Parameters for Random Forest

# %%
best_rf_params = grid_search_refined.best_params_

best_rf = RandomForestClassifier(**best_rf_params, random_state=42)
best_rf.fit(X_train, y_train)

y_pred_rf = best_rf.predict(X_test)

print("Best Parameters for Random Forest:", best_rf_params)
print("Accuracy for Tuned Random Forest:", accuracy_score(y_test, y_pred_rf))
print("Evaluation Metrics for Tuned Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=target_names, zero_division=0))

# %%
y_train_pred = best_rf.predict(X_train)
print(f"Fine-Tuned Random Forest Accuracy on Training Set: {accuracy_score(y_train, y_train_pred)}")
print("Fine-Tuned Random Forest Evaluation Metrics on Training Set:")
print(classification_report(y_train, y_train_pred, target_names=target_names, zero_division=0))

# %% [markdown]
# ### Visualize Accuracy for Different Parameter Combinations (Random Forest)

# %%
import matplotlib.pyplot as plt
import numpy as np

results = grid_search_refined.cv_results_

param_combinations = results['params']
mean_train_scores = results['mean_train_score'] if 'mean_train_score' in results else results['mean_test_score']
mean_test_scores = results['mean_test_score']
std_test_scores = results['std_test_score']

x_labels = [
    f"n_estimators={params['n_estimators']}, split={params['min_samples_split']}, depth={params['max_depth']}, crit={params['criterion']}"
    for params in param_combinations
]

sorted_indices = np.argsort(mean_test_scores)
x_labels = np.array(x_labels)[sorted_indices]
mean_train_scores = np.array(mean_train_scores)[sorted_indices]
mean_test_scores = np.array(mean_test_scores)[sorted_indices]

plt.figure(figsize=(15, 7))
x = np.arange(len(x_labels))

plt.plot(x, mean_test_scores, label="Validation Accuracy", marker='o')

plt.xticks(x, x_labels, rotation=90)
plt.xlabel("Parameter Combinations")
plt.ylabel("Accuracy")
plt.title("Accuracy for Different Parameter Combinations (Random Forest)")
plt.legend()
plt.tight_layout()

plt.show()


# %% [markdown]
# ### Testing a Certain Random Forest Model with Specific Hyperparameters
# Decided to use the hyperparameters below for Random Forest as a 'test'. Configured to these hyperparameters after seeing the accuracy for different hyperparameter combinations.

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

rf = RandomForestClassifier(
    n_estimators=275,
    max_depth=25,
    min_samples_split=2,
    random_state=42,
    criterion='gini'
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))


# %%
len(X_train[0])

# %% [markdown]
# ## Task 2: Recommender Systems Using Item2Vec

# %%
df.head(5)

# %%
df['category'].unique()

# %%
df['item_id'].iloc[0]

# %%
from collections import defaultdict

user_item_sequences = defaultdict(list)
for _, row in df.iterrows():
    user_item_sequences[row['user_id']].append(row['item_id'])

item_sequences = list(user_item_sequences.values())

# %%
from gensim.models import Word2Vec

item2vec_model = Word2Vec(
    sentences=item_sequences,
    vector_size=10,
    window=5,
    min_count=1,
    sg=1,
    workers=4
)

item_embedding = item2vec_model.wv['2260466']

similar_items = item2vec_model.wv.most_similar('2260466', topn=5)
print(similar_items)


# %%
# a dictionary mapping item_id to item_name
item_id_to_name = dict(zip(df['item_id'], df['category']))


# %%
len(item_id_to_name)

# %%
item_id_to_name['2517880']

# %% [markdown]
# ### Given a particular item specified by 'item_id', recommend similar items!

# %%
item = df['item_id'].iloc[0]

similar_items = item2vec_model.wv.most_similar(item, topn=5)

print(f"Current Item: {item_id_to_name[item]}\n")
print("Recommended Items:")
for similar_item, similarity in similar_items:
    item_name = item_id_to_name[similar_item]
    print(f"Item ID: {similar_item}, Name: {item_name}, Similarity: {similarity}")


# %% [markdown]
# ### Using Precision@K and and Recall@K to measure the quality of a recommendation

# %%
def precision_at_k(recommended_items, relevant_items, k, item_id_to_name):

    recommended_names = [item_id_to_name[i] for i in recommended_items[:k]]
    relevant_names = {item_id_to_name[i] for i in relevant_items if i in item_id_to_name}
    relevant_k = set(recommended_names) & relevant_names
    return len(relevant_k) / k

def recall_at_k(recommended_items, relevant_items, k, item_id_to_name):

    recommended_names = [item_id_to_name[i] for i in recommended_items[:k]]
    relevant_names = {item_id_to_name[i] for i in relevant_items if i in item_id_to_name}
    relevant_k = set(recommended_names) & relevant_names
    return len(relevant_k) / len(relevant_names) if relevant_names else 0


# %% [markdown]
# ### Example Use of the Recommender System (documented with comments)

# %%
# Example for the first item in the dataset
item = str(df['item_id'].iloc[0])

# get recommendations using Item2Vec
similar_items = item2vec_model.wv.most_similar(item, topn=5)

# create a list of recommended item IDs
recommended_items = [similar_item for similar_item, _ in similar_items]

# retrieve relevant items for the user who interacted with the current item
user_id = df.loc[df['item_id'] == item, 'user_id'].values[0]
relevant_items = set(df.loc[df['user_id'] == user_id, 'item_id'])

# print the current item and its recommendations
print(f"Current Item: {item_id_to_name[item]}\n")
print("Recommended Items:")
for similar_item, similarity in similar_items:
    print(f"{item_id_to_name[similar_item]} (ID: {similar_item}, Similarity: {similarity:.4f})")

print("\nRelevant Items:")
for relevant_item in relevant_items:
    print(item_id_to_name[relevant_item])

# calculate Precision@K and Recall@K (our evaluation metrics for the recommender system)
k = 5
precision = precision_at_k(recommended_items, relevant_items, k, item_id_to_name)
recall = recall_at_k(recommended_items, relevant_items, k, item_id_to_name)

print(f"\nPrecision@{k}: {precision:.4f}")
print(f"Recall@{k}: {recall:.4f}")


# %%
for i in recommended_items:
    print(item_id_to_name[i])

# %%
for i in relevant_items:
    print(item_id_to_name[i])

# %% [markdown]
# ### Function that calculates the average precision and recall @ K, across the entire dataset.

# %%
# Function to calculate average Precision@K and Recall@K
def average_precision_recall_at_k(df, item2vec_model, k, item_id_to_name):

    precision_list = []
    recall_list = []
    
    for user_id in df['user_id'].unique():
        user_items = df[df['user_id'] == user_id]['item_id'].unique()
        
        for item in user_items:
            item = str(item)
            
            if item not in item2vec_model.wv:
                continue
            
            similar_items = item2vec_model.wv.most_similar(item, topn=k)
            recommended_items = [similar_item for similar_item, _ in similar_items]
            
            relevant_items = set(user_items)
            
            precision = precision_at_k(recommended_items, relevant_items, k, item_id_to_name)
            recall = recall_at_k(recommended_items, relevant_items, k, item_id_to_name)
            
            precision_list.append(precision)
            recall_list.append(recall)
    
    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0
    
    return avg_precision, avg_recall

# usage
k = 5
avg_precision, avg_recall = average_precision_recall_at_k(df, item2vec_model, k, item_id_to_name)

# results of above usage (print)
print(f"Average Precision@{k}: {avg_precision:.4f}")
print(f"Average Recall@{k}: {avg_recall:.4f}")



