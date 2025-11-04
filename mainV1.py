#############################################################################
##                                                                         ##
## THIS IS A HEART DISEASE PREDICTION SYSTEM THAT ESTIMATES WHETHER A      ##
## PATIENT HAS HEART DISEASE BASED ON MACHINE LEARNING ANALYSIS OF 303     ##
## CASES.                                                                  ##
##                                                                         ##
## THE MODEL USED IS LOGISTIC REGRESSION.                                  ##
##                                                                         ##
## THE TRAINING ACCURACY SCORE IS 85.12%, AND THE TESTING ACCURACY SCORE   ##
## IS 81.97%, WHICH IS CONSIDERED GOOD FOR A SMALL DATASET LIKE THIS ONE.  ##
## IN GENERAL, ACCURACY ABOVE 75% IS A SOLID RESULT IN THIS CONTEXT.       ##
##                                                                         ##
## IMPORTANT: THIS CODE IS FOR EDUCATIONAL PURPOSES ONLY. I AM NOT A       ##
## MEDICAL PROFESSIONAL, AND THE RESULTS FROM THIS SYSTEM SHOULD NOT BE    ##
## USED AS A BASIS FOR DIAGNOSIS OR MEDICAL DECISIONS. PLEASE CONSULT A    ##
## HEALTHCARE PROVIDER FOR PROFESSIONAL MEDICAL ADVICE.                    ##
##                                                                         ##
## CREATED BY YOUNES AMEZIANE                                              ##
##                                                                         ##
#############################################################################



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading thr csv data to pandas dataframe
heart_data = pd.read_csv('data.csv')


X = heart_data.drop(columns='target', axis=1) #store all the features here
Y = heart_data['target'] #store all the targets here

#splitting the data in testing and training data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y, random_state = 2)

#model I'm using
model = LogisticRegression()

#train the model
model.fit(X_train, Y_train)

#model evaluation using acuracy score
    #accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) * 100
print(f'Accuracy score on training data: {training_data_accuracy:.2f}%')

    #accuracy on test data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test) * 100
print(f'Accuracy score on testing data: {testing_data_accuracy:.2f}%')

#building a predictive system

input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
