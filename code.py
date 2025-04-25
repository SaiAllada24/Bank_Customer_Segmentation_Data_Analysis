#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px

#Loading dataset
bank=pd.read_csv('bank_transactions.csv')

#Seeing the basic schema and the first 5 rows
bank.head()

#Seeing the statistics and relevant info of the dataset
bank.info()
bank.describe()

#We can see there are null values so need to deal with them 
#Another observation is in the CustomerDOB the dates are in different formats so need to make them uniform


bank['CustomerDOB']=pd.to_datetime(bank['CustomerDOB'],errors='coerce', dayfirst=True)

print("Null values in datasets are:", bank.isnull().sum() )

#Dealing with Nulls
#Since there are only 3397 nulls in CustomerDOB which is 0.32% of total entries we can afford to drop them as it wouldn't make much of a difference to our analysis
#Similarly with CustLocation there are only 151 nulls so we can drop or impute with unknown as well
bank.dropna(subset=['CustomerDOB', 'CustLocation'], inplace=True)
#CustAccountBalance - there are 2369 nulls but it is an important metric for our analysis so we'll fill it with median balance or treat missing balances as 0 reflecting inactive accounts
bank['CustAccountBalance'].fillna(bank['CustAccountBalance'].median(), inplace=True)
#Dealing with nulls in CustGender - filling with mode values
bank['CustGender'].fillna(bank['CustGender'].mode()[0], inplace=True)


# Fill missing CustomerDOB with median age of the location
bank['CustomerDOB'] = pd.to_datetime(bank['CustomerDOB'], errors='coerce')

# Calculate the median age for each location
current_year = pd.Timestamp('today').year
bank['Age'] = current_year - bank['CustomerDOB'].dt.year
location_median_age = bank.groupby('CustLocation')['Age'].median()

# Impute missing CustomerDOB based on location's median age
bank['Age'] = bank.apply(lambda x: location_median_age[x['CustLocation']] if pd.isnull(x['Age']) else x['Age'], axis=1)

from datetime import datetime

# Calculate age from DOB
bank['Age'] = (datetime.now() - pd.to_datetime(bank['CustomerDOB'])).dt.days // 365


# Extract additional features from TransactionDate and TransactionTime (e.g., day of the week, month).
bank['TransactionDate'] = pd.to_datetime(bank['TransactionDate'])
bank['TransactionDay'] = bank['TransactionDate'].dt.day_name()
bank['TransactionMonth'] = bank['TransactionDate'].dt.month
bank['TransactionDay']=bank['TransactionDate'].dt.day
bank['TransactionWeekDay']=bank['TransactionDate'].dt.weekday

bank.head()


# Age Groups
age_bins = [18, 25, 35, 45, 55, 65, 75, 100]
age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
bank['AgeGroup'] = pd.cut(bank['Age'], bins=age_bins, labels=age_labels, right=False)

#Pad TransactionTime values to ensure correct length
bank['TransactionTime'] = bank['TransactionTime'].apply(lambda x: str(x).zfill(6))
#Convert TransactionTime to datetime format, handling errors
bank['TransactionTime'] = pd.to_datetime(bank['TransactionTime'], format='%H%M%S', errors='coerce').dt.time
#Define a function to categorize time into bins (Morning, Afternoon, Evening, Night)

def get_time_of_day(hour):
    if pd.isna(hour):
        return None  # Handle missing values
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'
bank['TimeOfDay'] = bank['TransactionTime'].apply(lambda x: get_time_of_day(x.hour if pd.notna(x) else None))

bank_cleaned = bank[(bank['Age'] > 0) & (bank['Age'] <= 100)]

#Analysis based on age
#Age vs Transaction Amount
Age_Transaction=px.scatter(data_frame=bank_cleaned,x='Age',y='TransactionAmount (INR)')
Age_Transaction.show()


#K means clusters for analysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Select numerical features for clustering
X = bank[['Age', 'TransactionAmount (INR)', 'CustAccountBalance']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
bank['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters
plt.figure(figsize=(10,6))
sns.scatterplot(x='Age', y='TransactionAmount (INR)', hue='Cluster', data=bank, palette='viridis')
plt.title('Customer Segmentation Clusters')
plt.show()


#Age vs Customer Balance
Age_CustomerBalance=px.histogram(data_frame=bank_cleaned,x='Age',y='CustAccountBalance')
Age_CustomerBalance.show()


#Customer Gender analysis
gender_count=bank['CustGender'].value_counts()
print(gender_count)

#Customer Gender vs Balance
Gender_CustBalance=px.histogram(data_frame=bank_cleaned,x='CustGender',y='CustAccountBalance')
Gender_CustBalance.show()

#Customer Gender vs Transaction amount
Gender_Transaction=px.histogram(data_frame=bank_cleaned,x='CustGender',y='TransactionAmount (INR)')
Gender_Transaction.show()

#Transaction amount analysis
#Transaction amount vs location

# Count of transactions per location
location_counts = bank['CustLocation'].value_counts().reset_index()
location_counts.columns = ['CustLocation', 'Count']  # Rename columns for clarity

# Bar plot for transaction count per location
location_transactions = px.bar(location_counts, x='Count', y='CustLocation', 
                               title='Count of Transactions per Location',
                               labels={'CustLocation': 'Customer Location', 'Count': 'Number of Transactions'})
location_transactions.show()


#Transaction time analysis
bank.groupby('TimeOfDay')['TransactionAmount (INR)'].mean().plot(kind='bar', title='Average Transaction Amount by Time of Day')


#Plotting correlation matrix
corr_matrix = bank[['Age', 'CustAccountBalance', 'TransactionAmount (INR)']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Customer Features')
plt.show()

bank.groupby('TimeOfDay')[['CustAccountBalance', 'TransactionAmount (INR)']].corr()

#Age based categorization
age_bins = [18, 25, 35, 45, 60, 100]
age_labels = ['18-25', '26-35', '36-45', '46-60', '60+']
bank['AgeGroup'] = pd.cut(bank['Age'], bins=age_bins, labels=age_labels)


#Statistical Analysis - Chi Square test
from scipy.stats import chi2_contingency

# Cross-tabulation of time of day and customer location
contingency_table = pd.crosstab(bank['TimeOfDay'], bank['CustLocation'])

# Perform Chi-Square test
stat, p, dof, expected = chi2_contingency(contingency_table)

if p < 0.05:
    print("There is a significant relationship between time of day and location.")
else:
    print("No significant relationship between time of day and location.")


#Gender analysis
# Binning Transaction Amount into categories
bins = [0, 500, 5000, float('inf')]  # You can adjust these thresholds based on your data distribution
labels = ['Low', 'Medium', 'High']
bank['TransactionCategory'] = pd.cut(bank['TransactionAmount (INR)'], bins=bins, labels=labels)

# Contingency table for Gender vs. Transaction Amount
gender_vs_transaction = pd.crosstab(bank['CustGender'], bank['TransactionCategory'])
print(gender_vs_transaction)


# Contingency table for Location vs. Transaction Amount
location_vs_transaction = pd.crosstab(bank['CustLocation'], bank['TransactionCategory'])
print(location_vs_transaction)

from scipy.stats import chi2_contingency

# Chi-square test for Gender vs. Transaction Amount
chi2_gender, p_gender, dof_gender, expected_gender = chi2_contingency(gender_vs_transaction)
print(f"Chi-square statistic (Gender): {chi2_gender}, p-value: {p_gender}")

# Chi-square test for Location vs. Transaction Amount
chi2_location, p_location, dof_location, expected_location = chi2_contingency(location_vs_transaction)
print(f"Chi-square statistic (Location): {chi2_location}, p-value: {p_location}")

# Feature-target split
X = bank.drop(columns=['TransactionAmount (INR)'])
y = bank['TransactionAmount (INR)']

# Encoding categorical variables
le = LabelEncoder()
categorical_cols = ['CustGender', 'CustLocation', 'CustomerID']
for col in categorical_cols:
    bank[col] = le.fit_transform(bank[col].astype(str))

# Convert remaining non-numeric columns to strings and encode
for col in bank.select_dtypes(include=['object']).columns:
    bank[col] = le.fit_transform(bank[col].astype(str))

from sklearn.model_selection import train_test_split
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Ensure X_train and X_test are DataFrames before scaling
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# Scaling numerical features
from sklearn.preprocessing import LabelEncoder,StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')





