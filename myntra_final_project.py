
# %%
# 1. Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
#%%
# 2. Load and Clean Data
def load_and_clean_data(filepath, nrows=20000):
    df = pd.read_csv(filepath, nrows=nrows)
    df['DiscountOffer_clean'] = df['DiscountOffer'].str.extract(r'(\d+)', expand=False).astype(float)
    df_clean = df.dropna(subset=[
        'DiscountOffer_clean', 'DiscountPrice (in Rs)', 'OriginalPrice (in Rs)', 'Ratings', 'Reviews'
    ])
    df_clean = df_clean[df_clean['DiscountOffer_clean'].between(0, 100)].reset_index(drop=True)
    df_clean['discount_bins'] = pd.cut(
        df_clean['DiscountOffer_clean'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['0-20%', '21-40%', '41-60%', '61-80%', '81-100%']
    )
    return df_clean

df_clean = load_and_clean_data("Myntra Fasion Clothing.csv")
#Reading the first 20,000 rows, extracting numerical discount values from the DiscountOffer column, removing rows with missing or invalid entries in key fields (DiscountPrice, OriginalPrice, Ratings, Reviews), filtering discounts to a realistic range (0–100%), and categorizing them into discount bins for analysis.
#%%
# 3. Target Variable for Classification
df_clean['Rating_Class'] = np.where(df_clean['Ratings'] >= 4.0, 1, 0)
#For each row in the dataset, if the Ratings value is greater than or equal to 4.0, it assigns a 1 (High Rating) to the new column Rating_Class; otherwise, it assigns a 0 (Low Rating)
#%%
# 4. Exploratory Data Analysis
# Distribution of Discount Percentages
sns.histplot(df_clean['DiscountOffer_clean'], bins=30, kde=True)
plt.title("Distribution of Discount Percentages")
plt.show()
#The distribution of discount percentages in the Myntra dataset reveals that most discounts fall between 40% and 70%, with a prominent peak around the 55–60% range—indicating this is the most common discount bracket on the platform. The overall shape of the distribution is slightly right-skewed, meaning very high discounts (above 80%) are relatively rare. On the other end, products with low discounts (less than 20%) are also uncommon, suggesting that Myntra generally favors more aggressive discounting strategies. This could imply that lower-discount products are either not frequently offered or are not as prominently featured in listings.

# Discount vs Ratings Scatter Plot
sns.boxplot(x='discount_bins', y='Ratings', data=df_clean)
plt.title("Ratings by Discount Bracket")
plt.show()
#The boxplot shows that while the average customer rating remains fairly consistent across discount levels, deeper discount brackets—especially 81–100%—exhibit slightly lower median ratings and a wider spread. This may reflect perceived lower product quality or value at extreme discounts. Products in the 0–40% range tend to receive more consistent and higher ratings.

# Ratings Across Discount Bins  Correlation Heatmap
sns.heatmap(df_clean[['DiscountOffer_clean', 'Ratings', 'Reviews', 'OriginalPrice (in Rs)', 'DiscountPrice (in Rs)']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()
#The heatmap shows that discounts have a slightly negative correlation with ratings, supporting our hypothesis that deeper discounts may slightly reduce perceived product value. There’s a strong correlation between original and discounted prices, as expected. However, reviews and ratings show little correlation, suggesting that higher engagement doesn’t necessarily mean higher satisfaction.

#%%
# 5. Statistical Testing
# ANOVA: Ratings across discount brackets
#To check if there’s a statistically significant difference in average product ratings across different discount brackets.
groups = [group['Ratings'] for _, group in df_clean.groupby('discount_bins')]
f_stat, p_anova = f_oneway(*groups)
print(f"ANOVA: F = {f_stat:.2f}, p = {p_anova:.5f}")


# T-Test: Low vs High Discount Ratings
low_discount = df_clean[df_clean['DiscountOffer_clean'] <= 40]['Ratings']
high_discount = df_clean[df_clean['DiscountOffer_clean'] > 40]['Ratings']
t_stat, p_ttest = ttest_ind(low_discount, high_discount, equal_var=False)
print(f"T-Test: t = {t_stat:.2f}, p = {p_ttest:.5f}")
#The ANOVA test shows that average customer ratings significantly differ across discount brackets. Since the p-value is almost zero, we can confidently reject the null hypothesis (which says all groups have the same average rating). So, discount levels do have an impact on how customers rate products.
#%%
# 6.Cohort Analysis: Avg Ratings by Discount Bins
#To calculate and visualize the average customer rating for each discount bracket (cohort).

avg_ratings = df_clean.groupby('discount_bins')['Ratings'].mean()
avg_ratings.plot(kind='bar', title='Avg Ratings by Discount Bracket')
plt.ylabel('Average Rating')
plt.show()

#This bar chart shows that average product ratings remain relatively stable across discount brackets, ranging from ~4.1 to 4.3. While there is a slight dip in ratings at the highest discount level (81–100%), the differences are modest. This suggests that while extreme discounts may slightly impact customer perception, they do not drastically lower product ratings.

 #%%
# 7. Classification Modeling
# prepares the Myntra dataset for classification modeling by transforming both categorical and textual features into a machine learning–friendly format. It starts by encoding categorical variables such as brand, category, and gender using label encoding after replacing any missing values with "Unknown." Then, it processes the product descriptions using TF-IDF vectorization, extracting the top 100 most relevant terms as numerical features. These encoded and vectorized features are combined with numerical variables like discount percentage, original price, and discounted price to form the final feature matrix X. The target variable y is defined as Rating_Class, a binary indicator of whether a product received a high rating (≥ 4.0). Finally, the data is split into training and testing sets using an 80-20 ratio, setting the foundation for building and evaluating classification models.
# Encode categorical variables
categorical_cols = ['BrandName', 'Category', 'category_by_Gender']
df_encoded = df_clean[categorical_cols].fillna("Unknown").astype(str)
label_encoders = {col: LabelEncoder().fit(df_encoded[col]) for col in categorical_cols}
for col in categorical_cols:
    df_clean[col + '_enc'] = label_encoders[col].transform(df_encoded[col])

# TF-IDF on product descriptions
df_clean['Description'] = df_clean['Description'].fillna("")
tfidf = TfidfVectorizer(max_features=100)
description_tfidf = tfidf.fit_transform(df_clean['Description'])

# Final feature set
X = np.hstack([
    df_clean[['DiscountOffer_clean', 'OriginalPrice (in Rs)', 'DiscountPrice (in Rs)']].fillna(0).values,
    df_clean[['BrandName_enc', 'Category_enc', 'category_by_Gender_enc']].values,
    description_tfidf.toarray()
])
y = df_clean['Rating_Class'].values

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# Model 1: Logistic Regression
#This code trains and evaluates a Logistic Regression classification model on the prepared Myntra dataset. The model is initialized with a maximum of 500 iterations to ensure convergence. It is then trained using the X_train and y_train data. Once trained, the model predicts the class labels (y_pred_lr) on the test set X_test. Finally, it prints a classification report, which includes key performance metrics like precision, recall, F1-score, and accuracy. These metrics help assess how well the model distinguishes between high-rated and low-rated products, giving insight into its effectiveness in predicting customer satisfaction based on discount, price, brand, category, and description features.
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("🔹 Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))
#The model correctly predicted the rating class for 81% of the test data.
#The logistic regression model achieves an overall accuracy of 81%, with strong performance in predicting high-rated products (Precision: 0.82, Recall: 0.97). However, it struggles with low-rated predictions, showing the impact of class imbalance and suggesting the need for further model tuning or resampling."

#%%
# Model 2: Random Forest Classifier
#This code trains a Random Forest Classifier to predict whether a product will get a high rating (≥ 4.0) using features like discount, price, brand, and description text. It evaluates the model using a classification report to measure accuracy, precision, recall, and F1-score on the test set.
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("🔹 Random Forest Report:")
print(classification_report(y_test, y_pred_rf))
#The model performs very well in predicting high ratings but struggles to catch low-rated products due to class imbalance. It improves over logistic regression, particularly in detecting more low-rated items.
# %%
#This project aimed to analyze how product features such as discount percentage, pricing, brand, category, and description affect customer ratings on Myntra and to build a predictive model to classify whether a product would receive a high rating (≥ 4.0). 
# #Through EDA, statistical testing, and cohort analysis, we discovered subtle but significant relationships between discount levels and perceived product value.
#  We built and evaluated two classification models, Logistic Regression and Random Forest, achieving up to 82% accuracy in predicting high-rated products with strong performance in identifying positive customer feedback. 
# However, the models struggled to predict low-rated products due to class imbalance and the absence of deeper customer-related features like reviews or return rates. 
# Future improvements could include balancing techniques, richer feature sets, or ensemble models. 
# Despite these limitations, the project successfully operationalizes machine learning into a real-world e-commerce scenario and demonstrates how businesses can make data-informed decisions to optimize pricing and product presentation strategies.









