import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve

df = pd.read_csv('steam_top_games_2026.csv')


df['total_reviews'] = df['positive_reviews'] + df['negative_reviews']
df['approval_ratio'] = (df['positive_reviews'] / df['total_reviews']).fillna(0)
df['is_hit'] = ((df['approval_ratio'] >= 0.95) & (df['total_reviews'] >= 500)).astype(int)

df['ccu_intensity_log'] = np.log1p(df['peak_ccu'] / (df['total_reviews'] + 1))
df['price_tier'] = pd.cut(df['price_usd'], bins=[-1, 0, 5, 15, 30, 60, 1000], labels=[0, 1, 2, 3, 4, 5]).astype(int)
developer_avg = df.groupby('developer')['metacritic_score'].transform('mean')
df['metacritic_proxy'] = df['metacritic_score'].fillna(developer_avg).fillna(df['metacritic_score'].mean()).fillna(0)

genre_counts = df['genres'].str.split(', ').explode().value_counts()
def get_saturation(items_string, count_dict):
    if pd.isna(items_string): return 0
    items = items_string.split(', ')
    return np.mean([count_dict.get(item, 0) for item in items])

df['genre_saturation'] = df['genres'].apply(lambda x: get_saturation(x, genre_counts))
df['platform_count'] = df[['platforms_win', 'platforms_mac', 'platforms_linux']].sum(axis=1)
df['is_free_binary'] = df['is_free'].astype(int)
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce', format='mixed').dt.year
df['game_age'] = 2026 - df['release_year']

features = ['price_tier', 'dlc_count', 'achievements', 'ccu_intensity_log', 'genre_saturation', 'platform_count', 'metacritic_proxy', 'is_free_binary', 'game_age']
model_df = df[features + ['is_hit']].dropna()
X = model_df[features]
y = model_df['is_hit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample', random_state=42, max_depth=12, min_samples_leaf=2)
model.fit(X_train, y_train)

importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
print(importances.sort_values(by='Importance', ascending=False).to_string(index=False))

y_probs = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
optimal_threshold = thresholds[np.argmax(f1_scores)]

y_pred = (y_probs >= optimal_threshold).astype(int)
print(f"\nThreshold: {optimal_threshold:.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
