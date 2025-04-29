# Myntra Discount Impact Analysis

**Objective**: To analyze how discount percentages influence product ratings and review counts on Myntra, and explore whether this relationship differs across price tiers, gender-specific categories, and brands.
#%%
# Load required libraries
import pandas as pd
import numpy as np
#%%
# Load the data
df = pd.read_csv("Myntra Fasion Clothing.csv")
#%%
# 1. Clean 'DiscountOffer' column
# Remove '%' and non-numeric suffixes like ' OFF', ' Hurry*', etc.
df['DiscountOffer_clean'] = (
    df['DiscountOffer']
    .str.extract(r'(\d+)', expand=False)  # Extract only the number part
    .astype(float)  # Convert to numeric
)

# 2. Drop rows where key numerical fields are missing
df_clean = df.dropna(subset=[
    'DiscountOffer_clean',
    'DiscountPrice (in Rs)',
    'OriginalPrice (in Rs)',
    'Ratings',
    'Reviews'
])

# 3. Reset index
df_clean = df_clean.reset_index(drop=True)

# Optional: Check the cleaned dataset
print(df_clean[['DiscountOffer', 'DiscountOffer_clean', 'Ratings']].head())


# %%
#STEP 2: EDA – Exploring Discount Impact
#1. Distribution of Discounts
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(df_clean['DiscountOffer_clean'], bins=30, kde=True)
plt.title("Distribution of Discount Percentages")
plt.xlabel("Discount (%)")
plt.ylabel("Number of Products")
plt.show()


# %%
# 2. Discount vs Ratings (Scatterplot)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='DiscountOffer_clean', y='Ratings', data=df_clean, alpha=0.3)
plt.title("Discount % vs Ratings")
plt.xlabel("Discount (%)")
plt.ylabel("Ratings")
plt.show()

# %%
#🔹 3. Boxplot of Ratings Across Discount Brackets
# Create discount bins
df_clean['discount_bins'] = pd.cut(df_clean['DiscountOffer_clean'],
                                   bins=[0, 20, 40, 60, 80, 100],
                                   labels=['0-20%', '21-40%', '41-60%', '61-80%', '81-100%'])

plt.figure(figsize=(8, 5))
sns.boxplot(x='discount_bins', y='Ratings', data=df_clean)
plt.title("Ratings Across Discount Brackets")
plt.xlabel("Discount Bracket")
plt.ylabel("Ratings")
plt.show()

# %%
#🔹 4. Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df_clean[['DiscountOffer_clean', 'Ratings', 'Reviews', 'OriginalPrice (in Rs)', 'DiscountPrice (in Rs)']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# %%
#STEP 3: ANOVA – Discount Brackets vs Ratings
#🔹 1. Group Ratings by Discount Bins and Run ANOVA
from scipy import stats

# Create discount bins if not done already
df_clean['discount_bins'] = pd.cut(df_clean['DiscountOffer_clean'],
                                   bins=[0, 20, 40, 60, 80, 100],
                                   labels=['0-20%', '21-40%', '41-60%', '61-80%', '81-100%'])

# Create list of rating groups by discount bin
rating_groups = [group['Ratings'].dropna() for _, group in df_clean.groupby('discount_bins')]

# Run one-way ANOVA
f_stat, p_val = stats.f_oneway(*rating_groups)

print(f"F-statistic: {f_stat:.2f}")
print(f"p-value: {p_val:.5f}")

#The p-value is effectively zero, meaning the average ratings significantly differ across discount brackets. This suggests that discount levels do have a measurable impact on customer ratings.
#%%
# STEP 4: T-Test – Comparing Ratings for Low vs High Discounts
# 1. Run Independent T-test
from scipy.stats import ttest_ind

# Split data into two groups
low_discount = df_clean[df_clean['DiscountOffer_clean'] <= 40]['Ratings']
high_discount = df_clean[df_clean['DiscountOffer_clean'] > 40]['Ratings']

# Run independent t-test
t_stat, p_val = ttest_ind(low_discount.dropna(), high_discount.dropna(), equal_var=False)

print(f"T-statistic: {t_stat:.2f}")
print(f"p-value: {p_val:.5f}")
#There is a statistically significant difference in ratings between low-discount (≤40%) and high-discount (>40%) products.This reinforces the ANOVA result — discount levels do impact customer ratings.

# %%
#STEP 5.1: Linear Regression – Predict Ratings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Select features and target
features = ['DiscountOffer_clean', 'OriginalPrice (in Rs)', 'DiscountPrice (in Rs)']
X = df_clean[features]
y = df_clean['Ratings']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test)

# Evaluate
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression RMSE: {rmse_lr:.2f}")
print(f"Linear Regression R² Score: {r2_lr:.4f}")
#This suggests that a simple linear model isn't capturing much — possibly because ratings are influenced by non-numeric factors (e.g., product quality, style, brand reputation).
# %%
#Actual vs Predicted Plot (Linear Regression)
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings – Linear Regression")
plt.grid(True)
plt.tight_layout()
plt.show()



# %%
#🧠 STEP 5.2: Random Forest – Predict Ratings
from sklearn.ensemble import RandomForestRegressor

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest RMSE: {rmse_rf:.2f}")
print(f"Random Forest R² Score: {r2_rf:.4f}")
#The Random Forest model performs slightly better than Linear Regression — lower RMSE and higher R².

#But overall, it still doesn't explain much variance in the ratings (R² = 0.08).

#This suggests that numerical factors like discount and price alone aren’t strong predictors of ratings.

#Ratings likely depend more on subjective factors (fit, fabric, brand perception, etc.) not captured in this dataset.



# %%
#✅ 1. 📈 Actual vs Predicted Plot – Random Forest
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings – Random Forest")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
#🧠 STEP 5.3: XGBoost – Predict Ratings
from xgboost import XGBRegressor

# Train XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate
rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared=False)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost RMSE: {rmse_xgb:.2f}")
print(f"XGBoost R² Score: {r2_xgb:.4f}")
#Although XGBoost is more powerful, in this case, none of the models could explain much of the variation in customer ratings. This suggests:

#Discounts and price aren’t strong predictors of product ratings on their own.

#Ratings likely depend more on qualitative aspects like product feel, fit, look, fabric — which are not captured in this dataset.

#You can say:

#Statistical analysis (ANOVA, T-test) shows significant rating differences across discount levels.

#But prediction accuracy remains low, indicating other factors dominate consumer perception.


# %%
#📈 Actual vs Predicted Plot – XGBoost
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings – XGBoost")
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
#STEP 6: PCA – Dimensionality Reduction
# 1. Select & Scale Numeric Features

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Select numeric features
pca_features = ['DiscountOffer_clean', 'OriginalPrice (in Rs)', 'DiscountPrice (in Rs)', 'Ratings', 'Reviews']
df_pca = df_clean[pca_features].dropna()

# Standardize
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_pca)

#2. Apply PCA (2 Components)# %%
# Apply PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

# Create a DataFrame for visualization
pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])

#%%
# #3. Visualize PCA Result
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.4)
plt.title("PCA – Discount, Price, Ratings, Reviews")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Explained Variance by PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"Explained Variance by PC2: {pca.explained_variance_ratio_[1]:.2%}")

# %%
