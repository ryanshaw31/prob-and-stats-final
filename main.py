import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, confusion_matrix

# Last minute fix cause python didnt want to read the csv file
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'steam_top_games_2026.csv')

df = pd.read_csv(csv_path)

df['total_reviews'] = df['positive_reviews'] + df['negative_reviews']
df['approval_ratio'] = (df['positive_reviews'] / df['total_reviews']).fillna(0)
df['is_hit'] = ((df['approval_ratio'] >= 0.95) & (df['total_reviews'] >= 500)).astype(int)

df['ccu_intensity_raw'] = df['peak_ccu'] / (df['total_reviews'] + 1)
df['ccu_intensity_log'] = np.log1p(df['ccu_intensity_raw'])

df['price_tier'] = pd.cut(df['price_usd'], bins=[-1, 0, 5, 15, 30, 60, 1000], labels=[0, 1, 2, 3, 4, 5]).astype(int)

developer_avg = df.groupby('developer')['metacritic_score'].transform('mean')
df['metacritic_proxy'] = df['metacritic_score'].fillna(developer_avg).fillna(df['metacritic_score'].mean()).fillna(0)

genre_counts_series = df['genres'].str.split(', ').explode().value_counts()
def get_saturation(items_string, count_dict):
    if pd.isna(items_string): return 0
    items = items_string.split(', ')
    return np.mean([count_dict.get(item, 0) for item in items])

df['genre_saturation'] = df['genres'].apply(lambda x: get_saturation(x, genre_counts_series))

df['platform_count'] = df[['platforms_win', 'platforms_mac', 'platforms_linux']].sum(axis=1)
df['is_free_binary'] = df['is_free'].astype(int)
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce', format='mixed').dt.year
df['game_age'] = 2026 - df['release_year']

features_with = ['price_tier', 'dlc_count', 'achievements', 'ccu_intensity_log', 'genre_saturation', 'platform_count', 'metacritic_proxy', 'is_free_binary', 'game_age']
features_without = ['price_tier', 'dlc_count', 'achievements', 'ccu_intensity_log', 'genre_saturation', 'platform_count', 'is_free_binary', 'game_age']

def train_and_evaluate(feature_list):
    model_df = df[feature_list + ['is_hit']].dropna()
    X = model_df[feature_list]
    y = model_df['is_hit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample', random_state=42, max_depth=12, min_samples_leaf=2)
    rf.fit(X_train, y_train)
    probs = rf.predict_proba(X_test)[:, 1]
    p, r, t = precision_recall_curve(y_test, probs)
    return rf, X_test, y_test, probs, p, r

model_final, X_test_f, y_test_f, y_probs_f, p_with, r_with = train_and_evaluate(features_with)
_, _, _, _, p_without, r_without = train_and_evaluate(features_without)

opt_thresh = 0.3820
y_pred = (y_probs_f >= opt_thresh).astype(int)
importances = pd.DataFrame({'Feature': features_with, 'Importance': model_final.feature_importances_})
importances = importances.sort_values(by='Importance', ascending=False)

print("FEATURE IMPORTANCE (Final Model):")
print(importances.to_string(index=False))
print(f"\nThreshold: {opt_thresh:.4f}")
print(f"Accuracy: {accuracy_score(y_test_f, y_pred):.4f}")
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test_f, y_pred))

sns.set_theme(style="whitegrid")

fig1, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(df['ccu_intensity_raw'], bins=50, kde=True, ax=axes[0], color='tomato', edgecolor='black')
axes[0].set_title('Distribution of Raw CCU Intensity', fontsize=14)
axes[0].set_ylabel('Frequency (Count)')
sns.histplot(df['ccu_intensity_log'], bins=50, kde=True, ax=axes[1], color='skyblue', edgecolor='black')
axes[1].set_title('Log Transformation of Engagement', fontsize=14)
axes[1].set_ylabel('Frequency (Count)')
plt.tight_layout()

plt.figure(figsize=(10, 6))
plt.plot(r_with, p_with, label='Model WITH Metacritic Proxy', color='teal', linewidth=3)
plt.plot(r_without, p_without, label='Model WITHOUT Metacritic Proxy', color='darkorange', linestyle='--', linewidth=2)
plt.title('Impact of Metacritic Proxy on "Hit" Prediction Quality', fontsize=14)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.tight_layout()

genres_exploded = df['genres'].str.split(', ').explode()
genre_counts_df = genres_exploded.value_counts().reset_index()
genre_counts_df.columns = ['Genre', 'Frequency']
plt.figure(figsize=(12, 8))
sns.barplot(data=genre_counts_df, y='Genre', x='Frequency', hue='Genre', palette='viridis', legend=False)
plt.title('Frequency of Game Genres in Dataset', fontsize=14)
plt.xlabel('Frequency (Count)')
plt.ylabel('Genre')
plt.tight_layout()

plt.figure(figsize=(10, 6))
price_tier_counts = df['price_tier'].value_counts().sort_index().reset_index()
price_tier_counts.columns = ['Price Tier', 'Frequency']
tier_labels = ['Free', '$0.01-5', '$5-15', '$15-30', '$30-60', '$60+']
sns.barplot(data=price_tier_counts, x='Price Tier', y='Frequency', hue='Price Tier', palette='viridis', legend=False, edgecolor='black')
plt.xticks(ticks=range(6), labels=tier_labels)
plt.title('Frequency of Games per Price Tier', fontsize=14)
plt.xlabel('Price Tier Category')
plt.ylabel('Frequency (Count)')
plt.tight_layout()

plt.figure(figsize=(10, 6))
sns.barplot(data=importances, x='Importance', y='Feature', hue='Feature', palette='viridis_r', legend=False)
plt.title('Random Forest Feature Importance: Predicting "Overwhelmingly Positive" Status', fontsize=12)
plt.xlabel('Gini Importance (Reduction in Impurity)', fontsize=11)
plt.ylabel('Features', fontsize=11)
plt.tight_layout()

cm = confusion_matrix(y_test_f, y_pred)
plt.figure(figsize=(8, 6))
sns.set_theme(style="white")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Non-Hit', 'Predicted Hit'],
            yticklabels=['Actual Non-Hit', 'Actual Hit'])
plt.title('Confusion Matrix: Predicting "Overwhelmingly Positive" Customer Reviews', fontsize=14)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

plt.show()  