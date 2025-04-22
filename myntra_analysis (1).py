#%%

# # Load Libraries
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# %%
# Load Data & Clean
df = pd.read_csv("Myntra Fasion Clothing.csv")

df['DiscountOffer_clean'] = (
    df['DiscountOffer']
    .str.extract(r'(\d+)', expand=False)
    .astype(float)
)

df_clean = df.dropna(subset=[
    'DiscountOffer_clean', 'DiscountPrice (in Rs)', 'OriginalPrice (in Rs)', 'Ratings', 'Reviews'
]).reset_index(drop=True)

# Create discount bins
df_clean['discount_bins'] = pd.cut(df_clean['DiscountOffer_clean'],
                                   bins=[0, 20, 40, 60, 80, 100],
                                   labels=['0-20%', '21-40%', '41-60%', '61-80%', '81-100%'])

# %%
#EDA
# Discount Distribution
sns.histplot(df_clean['DiscountOffer_clean'], bins=30, kde=True)
plt.title("Distribution of Discount Percentages")
plt.show()

# Discount vs Ratings Scatter
sns.scatterplot(x='DiscountOffer_clean', y='Ratings', data=df_clean, alpha=0.3)
plt.title("Discount vs Ratings")
plt.show()

# Ratings Across Discount Bins (Boxplot)
sns.boxplot(x='discount_bins', y='Ratings', data=df_clean)
plt.title("Ratings Across Discount Brackets")
plt.show()

# Correlation Heatmap
sns.heatmap(df_clean[['DiscountOffer_clean', 'Ratings', 'Reviews', 'OriginalPrice (in Rs)', 'DiscountPrice (in Rs)']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# %%
#3. Statistical Testing Section
#ANOVA
rating_groups = [group['Ratings'].dropna() for _, group in df_clean.groupby('discount_bins')]
f_stat, p_val = stats.f_oneway(*rating_groups)
print(f"ANOVA F-statistic: {f_stat:.2f}, p-value: {p_val:.5f}")

# %%
#T-TEST
low_discount = df_clean[df_clean['DiscountOffer_clean'] <= 40]['Ratings']
high_discount = df_clean[df_clean['DiscountOffer_clean'] > 40]['Ratings']
t_stat, p_ttest = ttest_ind(low_discount, high_discount, equal_var=False)
print(f"T-Test: T-statistic = {t_stat:.2f}, p-value = {p_ttest:.5f}")

# %%
#CHISQUARE TEST
df_clean['rating_bucket'] = pd.cut(df_clean['Ratings'], bins=[0, 2.5, 3.5, 5], labels=['Low', 'Medium', 'High'])
ct_brand = pd.crosstab(df_clean['BrandName'], df_clean['rating_bucket'])
chi2, p_chi, _, _ = chi2_contingency(ct_brand)
print(f"Chi-square stat: {chi2:.2f}, p-value: {p_chi:.5f}")

# %%
#COHORT ANALYSIS
avg_rating_cohort = df_clean.groupby('discount_bins')['Ratings'].mean()
avg_reviews_cohort = df_clean.groupby('discount_bins')['Reviews'].mean()

# Plot both
avg_rating_cohort.plot(kind='bar', title='Avg Ratings per Discount Cohort')
plt.show()

avg_reviews_cohort.plot(kind='bar', title='Avg Reviews per Discount Cohort', color='orange')
plt.show()

# %%
#PREDECTIVE MODELING 
features = ['DiscountOffer_clean', 'OriginalPrice (in Rs)', 'DiscountPrice (in Rs)']
X = df_clean[features]
y = df_clean['Ratings']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Model
lr = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(f"Linear RMSE: {mean_squared_error(y_test, y_pred_lr, squared=False):.2f}, R²: {r2_score(y_test, y_pred_lr):.4f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Random Forest RMSE: {mean_squared_error(y_test, y_pred_rf, squared=False):.2f}, R²: {r2_score(y_test, y_pred_rf):.4f}")

# XGBoost
xgb = XGBRegressor(n_estimators=100).fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print(f"XGBoost RMSE: {mean_squared_error(y_test, y_pred_xgb, squared=False):.2f}, R²: {r2_score(y_test, y_pred_xgb):.4f}")

# %%
