import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('spam.csv')

# Split the data into features (email text) and labels (spam or not)
X = data['Message']
y = data['Category']

# Vectorize the email text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X, y)

# Make predictions on the training set
y_pred = classifier.predict(X)

# Calculate training accuracy
accuracy = accuracy_score(y, y_pred)
print("Training Accuracy:", (accuracy*100))

# Input an email message
email = input("Enter the email message: ")

# Vectorize the email message
email_vector = vectorizer.transform([email])

# Predict the category (spam or not)
prediction = classifier.predict(email_vector)

print(prediction) 