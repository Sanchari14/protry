from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import json
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# Global variable to store conversation context
conversation_context = {}
PREVIOUS_RESPONSE_KEY = "previous_responses"  # Key for storing previous responses
CONTEXT_TOPIC_KEY = "topic"  # Key for topic

# Load intents from JSON file
def load_intents(file_path="intents.json"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            intents = json.load(f)
        return intents
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {"intents": []}  # Return an empty structure to prevent errors

# Preprocess the text (tokenization, lemmatization)
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model, vectorizer

# Load model and vectorizer
intents = load_intents()
corpus, labels = get_training_data(intents)
model, vectorizer = train_model(corpus, labels)

# Predefined stress management techniques
stress_management_techniques = {
    "breathing_exercises": [
        "Deep breathing exercises can help calm your nervous system. Try the 4-7-8 technique: inhale for 4 seconds, hold for 7, and exhale for 8.",
        "Practice diaphragmatic breathing: place one hand on your chest and the other on your belly, and focus on expanding your belly as you inhale.",
    ],
    "mindfulness_meditation": [
        "Mindfulness meditation involves focusing on the present moment without judgment. There are many guided meditations available online.",
        "Try a simple body scan meditation: focus your attention on different parts of your body, noticing any sensations.",
    ],
    "progressive_muscle_relaxation": [
        "This technique involves tensing and releasing different muscle groups in your body. Start with your toes and work your way up.",
        "Progressive muscle relaxation can help you become more aware of tension in your body and how to release it.",
    ],
    "physical_activity": [
        "Engaging in regular physical activity can significantly reduce stress levels. Even a short walk can make a difference.",
        "Find an activity you enjoy, whether it's yoga, running, swimming, or dancing.",
    ],
    "time_management": [
        "Poor time management can contribute to stress. Try creating a schedule or to-do list to prioritize tasks.",
        "Break down large tasks into smaller, more manageable steps.",
    ],
    "social_support": [
        "Connecting with supportive people can help buffer the effects of stress. Talk to a friend, family member, or therapist.",
        "Engage in social activities you enjoy, such as spending time with loved ones or joining a club.",
    ],
    "positive_self_talk": [
        "Pay attention to your thoughts and challenge negative self-talk. Replace negative thoughts with more positive and realistic ones.",
        "Practice self-compassion: treat yourself with kindness and understanding, especially when you're going through a difficult time.",
    ],
    "healthy_diet": [
        "Eating a balanced diet can help your body cope with stress. Avoid excessive caffeine, alcohol, and processed foods.",
        "Focus on whole foods, such as fruits, vegetables, and whole grains.",
    ],
    "adequate_sleep": [
        "Getting enough sleep is crucial for both physical and mental health. Aim for 7-8 hours of quality sleep per night.",
        "Establish a regular sleep routine and create a relaxing bedtime environment.",
    ],
    "hobbies_and_interests": [
        "Make time for activities you enjoy, whether it's reading, listening to music, playing a musical instrument, or pursuing a creative hobby.",
        "Engaging in hobbies can provide a sense of purpose and accomplishment, and help you relax and recharge."
    ]
}

def get_response(intent_tag, intents, user_id):
    """
    Selects a response based on the intent tag, avoiding repetition, considering context,
    and suggesting stress management techniques.

    Args:
        intent_tag (str): The tag of the intent.
        intents (dict): The loaded intents data.
        user_id (str): The ID of the user.

    Returns:
        str: The selected response.
    """
    global conversation_context

    previous_responses = conversation_context.get(user_id, {}).get(PREVIOUS_RESPONSE_KEY, [])
    context_topic = conversation_context.get(user_id, {}).get(CONTEXT_TOPIC_KEY)

    for intent_data in intents["intents"]:
        if intent_data["tag"] == intent_tag:
            eligible_responses = [
                response for response in intent_data["responses"] if response not in previous_responses
            ]

            if eligible_responses:
                response = random.choice(eligible_responses)
            else:
                # If all responses have been used, reset and pick the first one
                conversation_context[user_id][PREVIOUS_RESPONSE_KEY] = []
                response = random.choice(intent_data["responses"])

            # Add a suggestion
            if context_topic == 'stress':
                technique = random.choice(list(stress_management_techniques.keys()))
                suggestion = random.choice(stress_management_techniques[technique])
                response += f" I suggest trying {technique}. {suggestion}"

            # Update previous responses
            conversation_context.setdefault(user_id, {}).setdefault(PREVIOUS_RESPONSE_KEY, []).append(response)
            conversation_context[user_id][PREVIOUS_RESPONSE_KEY] = conversation_context[user_id][PREVIOUS_RESPONSE_KEY][-5:]
            return response

    return "I'm sorry, I'm having trouble understanding. Can you please rephrase?"

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Invalid request. Missing "message" in JSON body.'}, 400)

        message = data['message']
        user_id = "default_user"  # Replace with actual user ID

        intents = load_intents()
        bow = create_bow(message, vectorizer)
        prediction = model.predict(bow)[0]

        response = get_response(prediction, intents, user_id)

        # Context Handling
        if prediction in ['stress', 'exam_stress', 'suicidal_thoughts', 'job_stress', 'salary_stress', 'parental_stress', 'reading_stress']:
            conversation_context[user_id] = {CONTEXT_TOPIC_KEY: 'stress'}
        elif prediction == 'greeting':
            conversation_context[user_id] = {}
        else:
            # Clear context, but provide a transition if coming from a stress context
            if conversation_context.get(user_id, {}).get(CONTEXT_TOPIC_KEY) == 'stress':
                response = "It's good we're talking. Remember, I'm here to support you. " + response
            conversation_context[user_id] = {}

        print(f"Selected response: {response}")
        return jsonify({'response': response}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}, 500)

if __name__ == '__main__':
    app.run(debug=True)
