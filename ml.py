#"C:\Users\Lenovo\Desktop\Diabates Model\diabetes.csv"

#Importing the dependencies

import numpy as np   #used to make arrays
import pandas as pd  #used for creating dataframe to structure the data
from sklearn.preprocessing import StandardScaler  #to standardized our data
from sklearn.model_selection import train_test_split  #to split the data for training and testing
from sklearn import svm #model
from sklearn.metrics import accuracy_score

#Data Collection and analysis

diabetes_dataset=pd.read_csv('diabetes.csv')

#print(diabetes_dataset.head())

print(diabetes_dataset.shape)

#getting the statistical measures
#print(diabetes_dataset.describe())
print(diabetes_dataset['Outcome'].value_counts())

#0-->non-diabetic
#1-->diabetic

print(diabetes_dataset.groupby('Outcome').mean())
#separating the data and labels

X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']
print(X)
print(Y)

#Data Standardization
scaler=StandardScaler()
scaler.fit(X)
standardized_data=scaler.transform(X)
print(standardized_data)

X=standardized_data
print(X)

#Train Test Split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2,stratify=Y,random_state=2)
print(X.shape, X_train.shape,X_test.shape)

#Training the model

classifier=svm.SVC(kernel='linear')

#training the support vectore machine classifier
classifier.fit(X_train,Y_train)

#Model Evaluation
#Accuracy Score

#accuracy score on the training data
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(Y_train,X_train_prediction)

print('Accuracy score of the training data : ',training_data_accuracy)

X_test_prediction=classifier.predict(X_test)
testing_data_accuracy=accuracy_score(Y_test,X_test_prediction)
print('Accuracy score of the testing data : ',testing_data_accuracy)

#making  a predictive system
input_data=(6,148,72,35,0,33.6,0.627,50)

#changing the input_data to numpy array
input_data_as_numpy_array = np.asanyarray(input_data)

#reshape the array as we are predicting for one data
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

#Standardized the input data
std_data=scaler.transform(input_data_reshaped)
print(std_data)

prediction=classifier.predict(std_data)
print(prediction)

if (prediction[0]==0):
      print("The Person is not diabetic")
else:
   print("The person is diabetic")


   import pickle

# Save model
with open('diabetes_model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

# Save scaler
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
