# %%
import gzip
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = []
file_path = "renttherunway_final_data.json.gz"

# Open data 
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding line: {e}")

df = pd.DataFrame(data)
df.head()

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

# %%
df[['fit', 'user_id', 'item_id', 'weight', 'rating', 'rented for', 'body type', 'category', 'height (in)', 'size', 'age']].head()

# ## Data Visualization

# %%
fit_counts = df["fit"].value_counts()
# Pie chart showing the distribution of the "fit" column
plt.figure(figsize=(8, 8))
fit_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
plt.title("Distribution of 'fit'")
plt.ylabel("")  # Removing the y-axis label for a cleaner pie chart
plt.show()

# %%
categorical_columns = ["rented for", "body type"]

# Bar plots for the distribution of the "rented for" and "body type" columns
fig, axes = plt.subplots(1, 2, figsize=(18, 5))
for i, col in enumerate(categorical_columns):
    sns.countplot(x=df[col], ax=axes[i])  # Countplot for each categorical column
    axes[i].set_title(f"Distribution of {col}")
    axes[i].tick_params(axis="x", rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()

# %%
numeric_columns = ["rating", "size", "age"]

# Histograms for the distribution of the "rating", "size", and "age" columns
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(numeric_columns):
    sns.histplot(df[col], bins=25, ax=axes[i])  # Histogram for each numeric column
    axes[i].set_title(f"Distribution of {col}")
plt.tight_layout()



