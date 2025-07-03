#  Preprocess Text and Labels

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
#  Load Dataset (we’ll use a small inline dataset for now)

import pandas as pd

# Sample dataset (literal)
data = {
    'text': [
        'Congratulations! You won a free ticket.',
        'Hi, are we meeting tomorrow?',
        'Win $1000 cash now!!!',
        'Hello, how are you?',
        'Urgent: Your loan is approved.',
        'Are you coming to the party?',
        'You have been selected for a free cruise!',
        'Don’t forget the meeting at 10am.',
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)
print(df)


# Convert text into numerical features using Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

# Target labels
y = df['label']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Logistic Regression Model


model = LogisticRegression()
model.fit(X_train, y_train)

#  Make Predictions and Evaluate


# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict New Message

msg = ["Congratulations, you've won a prize!"]
msg_vec = vectorizer.transform(msg)
prediction = model.predict(msg_vec)

print("Spam" if prediction[0] == 1 else "Not Spam")
