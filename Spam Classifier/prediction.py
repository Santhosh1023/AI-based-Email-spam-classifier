import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import tkinter as tk
from tkinter import *

# Load the dataset
data = pd.read_csv('spam.csv', encoding='ISO-8859-1')  # Replace 'your_dataset.csv' with your actual dataset path

# Data preprocessing function
def preprocess_text(text):
    # Example preprocessing steps: lowercasing and removing special characters
    text = text.lower()
    text = ''.join(e for e in text if (e.isalnum() or e.isspace()))
    return text

# Apply data preprocessing to the 'v2' (email_text) column
data['v2'] = data['v2'].apply(preprocess_text)

# Split the data into features (X) and labels (y)
X = data['v2']  # Email text
y = data['v1']  # Labels ('ham' or 'spam')

# Create a TfidfVectorizer for feature extraction
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Train the model
model = MultinomialNB()
model.fit(X_tfidf, y)

# Function to classify user input
def classify_email(user_input):
    user_input = preprocess_text(user_input)  # Apply preprocessing to user input
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    prediction = model.predict(user_input_tfidf)
    return prediction[0]

# Create a tkinter GUI window
window = tk.Tk()
window.title("Email Classifier")

text_label = tk.Label(window, text="Enter your message:")
text_label.pack()

text_entry = Text(window, height=5, width=40)
text_entry.pack()

result_label = tk.Label(window, text="")
result_label.pack()

def classify_button_click():
    user_input = text_entry.get("1.0", "end-1c")  # Get text from the Text widget
    result = classify_email(user_input)
    result_label.config(text=f"This message is classified as '{result}'.")
    text_entry.delete(1.0, END)  # Clear the text entry for the next input

classify_button = tk.Button(window, text="Classify", command=classify_button_click)
classify_button.pack()

window.mainloop()
