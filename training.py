import pandas as pd
from keras.models import Sequential
from keras.layers import *

training_data_df = pd.read_csv("training_scaled.csv")

X = training_data_df.drop('price', axis=1).values
Y = training_data_df[['price']].values

# Define the model
model = Sequential()
model.add(Dense(50, input_dim=8, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='SGD')

# Train the model
model.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2
)

# Load the separate test data set
test_data_df = pd.read_csv("testing_scaled.csv")

X_test = test_data_df.drop('price', axis=1).values
Y_test = test_data_df[['price']].values

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
X = pd.read_csv("proposed_product.csv").values

# Make a prediction with the neural network
prediction = model.predict(X)

# Grab just the first element of the first prediction (since that's the only have one)
prediction = prediction[0][0]

# Re-scale the data from the 0-to-1 range back to dollars
# These constants are from when the data was originally scaled down to the 0-to-1 range

print("You should sell this product at $: {}".format(1 / prediction))
