#-------------------------------------------------------------------------
# AUTHOR: Michael Acosta
# FILENAME: naive_bayes.py
# SPECIFICATION: Take in weather data, train a Naive Bayes model, and test 
#               on unlabeled data, showing predictions over a certain 
#               confidence.
# FOR: CS 4210- Assignment #2
# TIME SPENT: About 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

dbTraining = []
dbTest = []
predictions = []

#reading the training data in a csv file
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTraining.append(row)
        else:
            predictions.append(row)

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
Outlook = {
    "Sunny": 1,
    "Overcast": 2,
    "Rain": 3,
}
Temperature = {
    "Hot": 1,
    "Cool": 2,
    "Mild": 3,
}
Humidity = {
    "Normal": 1,
    "High": 2,
}
Wind = {
    "Weak": 1,
    "Strong": 2,
}
X = dbTraining.copy() #make a copy of file read in
for i, row in enumerate(dbTraining): 
    #transform features based on dictionary, ignoring day
    X[i] = [Outlook[row[1]], Temperature[row[2]], 
            Humidity[row[3]], Wind[row[4]]]

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
PlayTennis = {
    "No": 1,
    "Yes": 2,
}
Y = [0] * len(dbTraining) #create empty list
for j, row in enumerate(dbTraining):
    Y[j] = PlayTennis[row[5]] #transform label based on dictionary

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTest.append(row)

for k, row in enumerate(dbTest): 
    #transform features based on dictionary, ignoring day and PlayTennis class
    dbTest[k] = [row[0], Outlook[row[1]], Temperature[row[2]], 
            Humidity[row[3]], Wind[row[4]], row[5]]

#printing the header os the solution
predictions[0].append('Confidence')

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for instance in dbTest:
    results = clf.predict_proba([[instance[1], instance[2], instance[3], instance[4]]])[0]
    if results[0] > 0.75:
        instance[5] = 'No'
        instance.append(round(results[0], 3))
        predictions.append(instance)
    if results[1] > 0.75:
        instance[5] = 'Yes'
        instance.append(round(results[1], 3))
        predictions.append(instance)

for day in predictions:
   print(day)