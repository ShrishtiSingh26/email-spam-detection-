
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #FEATURE EXTRACTION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
raw_mail_data = pd.read_csv('/content/mail_data.csv') # loading the data from csv file
print(raw_mail_data)

mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'') #replace null values
mail_data.head()

mail_data.shape #no. of data

# label spam mail as 0;  ham mail as 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1
X = mail_data['Message']
Y = mail_data['Category']
print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3) #80%training 20% test
print(X.shape)
print(X_train.shape)
print(X_test.shape)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True) #feature extraction(converting in numerical values)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
print(X_train_features)

"""LOGISTIC REGRESSION"""

model = LogisticRegression()

model.fit(X_train_features, Y_train)

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data : ', accuracy_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data : ', accuracy_on_test_data)

input_mail = ["You have an important customer service announcement. Call FREEPHONE 0800 542 0825 now!"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')
