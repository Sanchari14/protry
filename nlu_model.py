import nltk
import json
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')

# Load intents from the JSON file
def load_intents(file_path="intents.json"):
    with open(file_path, 'r') as f:
        intents = json.load(f)
    return intents

# Preprocess the text (tokenization, lemmatization)
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]  # Lemmatize and lowercase
    return tokens

# Create a bag-of-words representation
def create_bow(text, vectorizer):
    tokens = preprocess_text(text)
    bow = vectorizer.transform([' '.join(tokens)]).toarray()
    return bow

# Get data for training
def get_training_data(intents):
    corpus = []
    labels = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            corpus.append(pattern)
            labels.append(intent['tag'])
    return corpus, labels

# Train the model
def train_model(corpus, labels):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #split
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model, vectorizer

def main():
    intents = load_intents()
    corpus, labels = get_training_data(intents)
    model, vectorizer = train_model(corpus, labels)

    # Save the model and vectorizer (optional, for later use)
    # import joblib
    # joblib.dump(model, 'chatbot_model.pkl')
    # joblib.dump(vectorizer, 'vectorizer.pkl')

    # Interactive loop (for testing from the command line)
    print("Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        bow = create_bow(user_input, vectorizer)
        prediction = model.predict(bow)[0]
        for intent in intents['intents']:
            if intent['tag'] == prediction:
                print("Chatbot:", random.choice(intent['responses']))
                break

if __name__ == "__main__":
    main()

