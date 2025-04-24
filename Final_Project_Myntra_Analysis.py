#%%
# Project Title: Decoding Product Perception: What Drives High Ratings on Myntra?
# Objective: To explore the patterns and predictors of product ratings, clean the data, validate insights with statistical tests,
# and build models to classify whether an item will be rated highly (4.0+) or not.

# Step 1: Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from math import pi
from sklearn.preprocessing import MinMaxScaler

#%%
# Step 2: Load and Clean Data

def load_and_clean_data(filepath, nrows=100000):
    df = pd.read_csv(filepath, nrows=nrows)

    # Extract numeric discount percent
    df['DiscountOffer_clean'] = df['DiscountOffer'].str.extract(r'(\d+)', expand=False).astype(float)

    # Drop rows with missing or noisy values
    df_clean = df.dropna(subset=[
        'DiscountOffer_clean', 'DiscountPrice (in Rs)', 'OriginalPrice (in Rs)', 'Ratings', 'Reviews'
    ])

    # Filter out invalid prices, ratings, and reviews
    df_clean = df_clean[
        (df_clean['OriginalPrice (in Rs)'] > 0) &
        (df_clean['DiscountPrice (in Rs)'] > 0) &
        (df_clean['Ratings'].between(1, 5)) &
        (df_clean['Reviews'] > 0)
    ]

    # Remove outliers using z-score for key numerical columns
    from scipy.stats import zscore
    z_cols = ['OriginalPrice (in Rs)', 'DiscountPrice (in Rs)', 'Reviews']
    df_clean = df_clean[(np.abs(zscore(df_clean[z_cols])) < 3).all(axis=1)]

    # Keep only discounts within a realistic range
    df_clean = df_clean[df_clean['DiscountOffer_clean'].between(0, 100)]

    # Reset index
    df_clean = df_clean.reset_index(drop=True)

    return df_clean

# Load dataset
df = load_and_clean_data("Myntra_Fasion_Clothing.csv")

# Create Rating_Class
mean_rating = df['Ratings'].mean()
df['Rating_Class'] = np.where(df['Ratings'] >= 4.0, 1, 0)

print(f"Mean Rating: {mean_rating:.2f}, Min: {df['Ratings'].min()}, Max: {df['Ratings'].max()}")
print("Class Distribution:")
print(df['Rating_Class'].value_counts())

#%%
# Step 3: Encode Categorical Features
le_brand = LabelEncoder()
df['Brand_encoded'] = le_brand.fit_transform(df['BrandName'])

le_gender = LabelEncoder()
df['Gender_encoded'] = le_gender.fit_transform(df['category_by_Gender'])

le_cat = LabelEncoder()
df['Category_encoded'] = le_cat.fit_transform(df['Category'])


#%%
# Step 4: EDA – Distribution Overview
sns.countplot(x='Rating_Class', data=df)
plt.title("Distribution of Rating Classes")
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df['Ratings'], bins=30, kde=True)
plt.title("Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

print("\nRatings Summary:")
print(df['Ratings'].describe())
print("\nVariance:", df['Ratings'].var())
quantiles = df['Ratings'].quantile([0.25, 0.5, 0.75])
print("\nQuantiles:")
print(quantiles)

# plt.figure(figsize=(8, 4))
# sns.histplot(df['DiscountOffer_clean'], bins=30, kde=True)
# plt.title("Discount Percentage Distribution")
# plt.xlabel("Discount %")
# plt.show()

# plt.figure(figsize=(8, 4))
# sns.boxplot(x='Rating_Class', y='OriginalPrice (in Rs)', data=df)
# plt.title("Original Price by Rating Class")
# plt.show()

# plt.figure(figsize=(8, 4))
# sns.violinplot(x='Rating_Class', y='Reviews', data=df)
# plt.title("Review Counts by Rating Class")
# plt.yscale("log")
# plt.show()

# Rating by Gender
plt.figure(figsize=(6,4))
sns.boxplot(x='category_by_Gender', y='Ratings', data=df)
plt.title("Ratings by Gender Category")
plt.xticks(rotation=45)
plt.show()

# Rating by Category
top_cats = df['Category'].value_counts().nlargest(10).index
df_topcats = df[df['Category'].isin(top_cats)]
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_topcats, x='Category', y='Ratings')
plt.title("Ratings by Top 10 Categories")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 4))
plt.scatter(
    df['DiscountOffer_clean'],
    df['Ratings'],
    s=8,               # small dots
    alpha=0.05,        # very transparent, will darken with density
    color='purple'     # or try 'black', 'blue', 'teal', etc.
)
plt.title("Discount % vs Rating (Density Visualized)")
plt.xlabel("Discount (%)")
plt.ylabel("Rating")
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
plt.tight_layout()
plt.show()

# # Reviews vs Ratings
# plt.figure(figsize=(8, 4))
# sns.scatterplot(data=df, x='Reviews', y='Ratings', alpha=0.2)
# plt.xscale("log")
# plt.title("Scatterplot: Reviews vs Ratings")
# plt.show()

#%%
# Step 5: Correlation Heatmap
numerics = ['OriginalPrice (in Rs)', 'DiscountPrice (in Rs)', 'DiscountOffer_clean', 'Ratings', 'Reviews']
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerics].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

#%%
# Hypothesis Testing - Refined Block

# H1a: Expensive products receive higher ratings
# H1b: Expensive products receive lower ratings
high_rating = df[df['Rating_Class'] == 1]['OriginalPrice (in Rs)']
low_rating = df[df['Rating_Class'] == 0]['OriginalPrice (in Rs)']

print("\nH1a: T-Test - Do highly rated products have higher prices?")
t_stat_1a, p_val_1a = ttest_ind(high_rating, low_rating, equal_var=False)
print(f"T-Test: t = {t_stat_1a:.2f}, p = {p_val_1a:.5f}")

print("\nH1b: Z-Test - Are higher prices significantly associated with high ratings?")
mean_diff = high_rating.mean() - low_rating.mean()
std_err = np.sqrt(high_rating.var()/len(high_rating) + low_rating.var()/len(low_rating))
z_stat_1b = mean_diff / std_err
from scipy.stats import norm
p_val_1b = 2 * (1 - norm.cdf(abs(z_stat_1b)))
print(f"Z-Test: z = {z_stat_1b:.2f}, p = {p_val_1b:.5f}")

print("\n---x---x---x---x---x---x---x---x---x---x---")

# H2a: Discount amount influences rating
# H2b: Higher discounts reduce average rating
df['Discount_bin'] = pd.cut(df['DiscountOffer_clean'], bins=[0, 20, 40, 60, 80, 100],
                             labels=['0-20%', '21-40%', '41-60%', '61-80%', '81-100%'])
grouped = [group['Ratings'] for _, group in df.groupby('Discount_bin')]

print("\nH2a: ANOVA - Do ratings differ across discount bins?")
f_stat_a, p_anova_a = f_oneway(*grouped)
print(f"ANOVA: F = {f_stat_a:.2f}, p = {p_anova_a:.5f}")

df['HighDiscount'] = df['DiscountOffer_clean'] >= 50
low_disc = df[df['HighDiscount'] == False]['Ratings']
high_disc = df[df['HighDiscount'] == True]['Ratings']
t_stat_b, p_val_b = ttest_ind(low_disc, high_disc, equal_var=False)
print("\nH2b: T-Test - Do higher discounts lower ratings?")
print(f"T-Test: t = {t_stat_b:.2f}, p = {p_val_b:.5f}")

print("\n---x---x---x---x---x---x---x---x---x---x---")
#%%
# Step 7: Classification Modeling
features = [
    'OriginalPrice (in Rs)', 'DiscountPrice (in Rs)', 'DiscountOffer_clean',
    'Brand_encoded', 'Gender_encoded', 'Category_encoded', 'Reviews'
]
X = df[features]
y = df['Rating_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

feat_names = X.columns

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

#Confustion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Oranges')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
coef = lr_model.coef_[0]
feat_importance = pd.Series(coef, index=feat_names).sort_values()
plt.figure(figsize=(8, 4))
sns.barplot(x=feat_importance.values, y=feat_importance.index)
plt.title("Logistic Regression Coefficient Importance")
plt.xlabel("Coefficient Value (Impact on Log-Odds)")
plt.tight_layout()
plt.show()

# ROC Curve
RocCurveDisplay.from_estimator(lr_model, X_test, y_test)
plt.title("ROC Curve - Logistic Regression")
plt.grid(True)
plt.show()

print("\nLogistic Regression Results:")
print(classification_report(y_test, y_pred_lr))


print("---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
importances = rf_model.feature_importances_
plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=feat_names)
plt.title("Random Forest Feature Importance")
plt.show()

# ROC Curve (Random Forest)
RocCurveDisplay.from_estimator(rf_model, X_test, y_test)
plt.title("ROC Curve - Random Forest")
plt.show()

print("\nRandom Forest Results:")
print(classification_report(y_test, y_pred_rf))


#%%
#%%
# Cluster Analysis - Ratings, Discounts, Reviews, Price, Gender (Enhanced Dimensions)
# Axes used: Ratings, DiscountOffer_clean, Gender_encoded, Reviews, OriginalPrice (in Rs)
cluster_features = ['Ratings', 'DiscountOffer_clean', 'Gender_encoded', 'Reviews', 'OriginalPrice (in Rs)']
cluster_data = df[cluster_features].dropna()
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_data)

# Elbow Plot to determine optimal k
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(cluster_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Plot for Optimal K")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# Final KMeans with K=4
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(cluster_scaled)
cluster_data['Cluster'] = clusters

# PCA for 2D projection
pca = PCA(n_components=2)
pca_result = pca.fit_transform(cluster_scaled)
cluster_data['PC1'] = pca_result[:, 0]
cluster_data['PC2'] = pca_result[:, 1]

plt.figure(figsize=(8, 6))
colors = sns.color_palette("tab10", 5)
for i in range(5):
    points = cluster_data[cluster_data['Cluster'] == i][['PC1', 'PC2']].values
    hull = ConvexHull(points)
    plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {i}", alpha=0.6, color=colors[i])
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], color=colors[i])
    polygon = Polygon(points[hull.vertices], alpha=0.2, color=colors[i])
    plt.gca().add_patch(polygon)

plt.title("KMeans Clusters with Convex Hulls (PCA View)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()

# Radar Plot - Combined, color-coded by cluster
cluster_means = cluster_data.groupby('Cluster')[cluster_features].mean()
scaler_radar = MinMaxScaler()
cluster_means_scaled = pd.DataFrame(scaler_radar.fit_transform(cluster_means), columns=cluster_features, index=cluster_means.index)

N = len(cluster_features)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i in cluster_means_scaled.index:
    values = cluster_means_scaled.loc[i].tolist()
    values += values[:1]
    ax.plot(angles, values, label=f"Cluster {i}", linewidth=2, color=colors[i])
    ax.fill(angles, values, alpha=0.25, color=colors[i])

ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(cluster_features, fontsize=11)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=9)
ax.set_title("Combined Radar Plot of Cluster Profiles", fontsize=15, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
plt.tight_layout()
plt.show()

#%%[markdown]
# ### 📊 Clustering Takeaways
# - 5 distinct clusters were identified using KMeans (with PCA for visualization).
# - Each cluster reveals a unique market segment:
#   - 🔵 **Cluster 0**: 💰 Highest original prices, moderate ratings and discounts — likely premium products with moderate appeal.
#   - 🟠 **Cluster 1**: ⭐ Extremely high ratings, lowest discounts, low prices — strong performers loved by customers.
#   - 🟢 **Cluster 2**: 📢 Highest review count, balanced gender and pricing — actively engaged, possibly trending mid-range products.
#   - 🔴 **Cluster 3**: 🔻 Low discount and low engagement — underperformers with little traction.
#   - 🟣 **Cluster 4**: 👨 Dominantly male-targeted, high discounts, low ratings — possibly fast-moving or overstocked items not perceived well.(with PCA for visualization).
# - Each cluster reveals a unique market segment:
#   - 🎯 Budget-friendly, top-rated products
#   - 🔻 Discount-heavy underperformers
#   - 🔄 Mid-market general performers
#   - 💰 Premium-priced, low-engagement items

#%% [markdown]
# ## 🧾 Final Summary
#
# After thorough exploration and modeling on over 30,000 fashion product entries from Myntra, we conclude:
#
# ### 🔍 Exploratory Insights
# - The average product rating is **~4.1**, with a tight distribution around it.
# - **Discounts and Reviews** are the most visually and statistically significant influencers.
# - Products with deeper discounts tend to **receive lower ratings**, on average.
# - **Women's products dominate** the higher-rating clusters.
#
# ### 🧪 Hypothesis Testing
# - We validated **4 hypotheses**, including:
#   - Expensive products being rated higher — *statistically significant with T-test and Z-test*.
#   - Discount bins affecting ratings — *confirmed via ANOVA and T-test*.
#
# ### 🧠 Modeling Insights
# - Both **Logistic Regression** and **Random Forest** models were built to classify high vs low-rated products.
# - Random Forest showed higher precision overall, but Logistic Regression provided interpretable coefficients.
# - Feature importance highlighted **Original Price, Brand, and Reviews** as top contributors.
#
# ### 📊 Clustering Takeaways
# - 4 KMeans clusters were formed on Ratings, Discounts, Reviews, Gender, and Price.
# - These revealed distinct user-product groups like:
#   - "Loyal fans of low-priced, high-rated items"
#   - "Discount-heavy but underperforming men's products"
#
# ### 🎯 Final Takeaway
# > **Customer perception on Myntra is influenced by more than just price. Reviews, brand reputation, gender orientation, and discount strategies collectively shape product success.**
#
# This analysis supports smarter inventory, marketing, and pricing strategies for e-commerce platforms.


# %%
