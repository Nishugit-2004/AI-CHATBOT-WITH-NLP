print("SCRIPT STARTED")


import nltk
import random
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Training data
intents = {
    "greeting": ["hello", "hi", "hey", "good morning"],
    "goodbye": ["bye", "see you", "goodbye"],
    "internship": ["what is internship", "tell me about internship", "internship details"],
    "ai": ["what is ai", "define artificial intelligence", "explain ai"],
    "ml": ["what is machine learning", "define ml"]
}

responses = {
    "greeting": ["Hello! How can I help you?", "Hi there!"],
    "goodbye": ["Goodbye!", "See you soon!"],
    "internship": ["An internship is a short-term work experience to gain practical knowledge."],
    "ai": ["AI is the simulation of human intelligence in machines."],
    "ml": ["Machine Learning is a subset of AI that allows systems to learn from data."]
}

# Prepare training data
X = []
y = []

for intent, patterns in intents.items():
    for pattern in patterns:
        X.append(pattern)
        y.append(intent)

# Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Chat function
def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)[0]
    return random.choice(responses[prediction])

# Chat loop
print("🤖 AI Chatbot is running! Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Bot:", response)
