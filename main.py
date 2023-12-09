#STEP 1:Importing libaries
import pandas as pd
import pickle
#STEP 2:EDA WITH DATASET
# Loading the dataset
df = pd.read_csv('ipl.csv')
# print(df.head())
#STEP 3:Removing unwanted columns
columns_to_remove=['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
df.drop(labels=columns_to_remove, axis=1, inplace=True)
# print(df['bat_team'].unique())
# we will only be taking current teams\
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
# print(df)
# Removing the first 5 overs data in every match
df = df[df['overs']>=5.0]
# print(df['bat_team'].unique())
# print(df['bowl_team'].unique())

# converting date cols from string to date time object
from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
print(df['date'].dtype)
# STEP 4:
# Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
print(encoded_df.head())
# Rearranging the columns
encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]
# Splitting the data into train and test set
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]
y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values
# X_train = All training features except the target column(runs) up to the year 2016
# X_test = All training features except the target column(runs) for the remaining years
# y_train = target column up to the year 2016
# y_test = target column for the remaining years.
print(X_train.shape , y_train.shape)
print(X_test.shape , y_test.shape)
# since the requirement of our date column is over so we can drop it
# dropping date column
X_train.drop(labels = 'date', axis = True, inplace = True)
X_test.drop(labels = 'date', axis = True, inplace = True)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train , y_train)
prediction = reg.predict(X_test)
# print(prediction)

# Custom accuracy metric (percentage)
def custom_accuracy(y_true, y_pred, tolerance_percentage=10):
    # Calculate absolute percentage error
    abs_percentage_error = (abs(y_true - y_pred) / y_true) * 100

    # Count the number of predictions within the specified tolerance
    correct_predictions = sum(abs_percentage_error <= tolerance_percentage)

    # Calculate the percentage of correct predictions
    accuracy_percentage = (correct_predictions / len(y_true)) * 100

    return accuracy_percentage

# Calculate custom accuracy
accuracy = custom_accuracy(y_test, prediction)

# print(f'Custom accuracy: {accuracy:.2f}%')
# Creating a pickle file for the classifier
filename = 'first-innings-score-lr-model.pkl'
pickle.dump(reg, open(filename, 'wb'))
