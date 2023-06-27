import pandas as pd
import requests
import json
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import streamlit as st 

# Specify the API endpoint you want to access
url = 'https://pandemicdatalake.blob.core.windows.net/public/curated/covid-19/bing_covid-19_data/latest/bing_covid-19_data.json'
# Make a GET request to the API
response = requests.get(url)
# Check if the request was successful (status code 200)
if response.status_code == 200:
    data = response.json()
    # process the data
    covid_data = pd.DataFrame.from_dict(data)
    covid_data.head()
    # Remove null values
    covid_data = covid_data.dropna(axis=0)
    # Drop the extra column 
    labels = ['id', 'latitude', 'longitude', 'iso2', 'iso3', 'admin_region_2', 'load_time']
    covid_data = covid_data.drop(labels=labels, axis=1)
    # fix updated data type as datetime
    covid_data['updated'] = covid_data['updated'].astype('datetime64[ns]')
    # data info
    covid_data.info()
    # Select the numerical columns for normalization and scaling
    numerical_columns = ['confirmed', 'deaths', 'recovered']
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()
    # Normalize and scale the numerical columns
    covid_data[numerical_columns] = scaler.fit_transform(covid_data[numerical_columns])

else:
    # Request was not successful, handle the error
    print("Error:", response.status_code)

# Plot a histogram
plt.hist(covid_data['confirmed'], bins=10)
plt.xlabel('confirmed_case')
plt.ylabel('Frequency')
plt.title('Histogram of confirmed_case')
plt.show()
# Plot a distribution plot
sns.displot(covid_data, x=covid_data['updated'], y=covid_data['confirmed'], hue=covid_data['country_region'])
plt.show()
# Plot a scatter plot
sns.scatterplot(data=covid_data, x='confirmed', y='deaths')
plt.show()

sns.scatterplot(data=covid_data, x='confirmed_change', y='deaths_change')
plt.show()

sns.scatterplot(data=covid_data, x='confirmed_change', y='recovered_change')
plt.show()

# Calculate the correlation matrix
corr_matrix = covid_data.corr()

# Plot a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Drop the column with problematic values
labels = ['updated', 'admin_region_1', 'iso_subdivision', 'country_region']
covid_data = covid_data.drop(labels=labels, axis=1)

# Define the features and the target variable
X = covid_data.drop('deaths_change', axis=1)  # the features
y = covid_data['deaths_change']  # the target variable

# Split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying DecisionTreeRegressor to predict deaths change
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

# Evaluate DecisionTreeRegressor model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('mean_squared_error: ', mse)
print('mean_absolute_error: ', mae)
print('r2_score: ' , r2)
# Perform cross-validation with 5 folds
scores = cross_val_score(decision_tree, X_train, y_train, cv=5)
average_score = scores.mean()
# Select best parameter by performing GridSearchCV
param_grid = {
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(decision_tree, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(best_params)
y_pred = grid_search.predict(X_test)

# Create a streamlit App
st.title('Prédiction du nombre de cas COVID-19')

# Create input buton to introduce features (independante variables)
confirmed = st.number_input('Numbre of confirmed', value=0)
recovered = st.number_input('Numbre of recovered', value=0)
Confirmed_change = st.number_input('Confirmed_change', value=0)
recovered_change = st.number_input('recovered_change', value=0)


# Création d'un bouton pour effectuer la prédiction
if st.button('Prédire'):
    # Création d'un DataFrame avec les données saisies par l'utilisateur
    input_data = pd.DataFrame({
        'confirmed': [confirmed],
        'recovered': [recovered],
        'Confirmed_change': [Confirmed_change],
        'recovered_change': [recovered_change],
    })

    # Normalisation des données saisies par l'utilisateur
    input_data_scaled = scaler.transform(input_data)

    # Faire une prédiction avec le modèle entraîné
    prediction = decision_tree.predict(input_data_scaled)

    # Affichage la prédiction
    st.write(f'Prédiction du nombre de cas : {prediction[0]:.0f}')
