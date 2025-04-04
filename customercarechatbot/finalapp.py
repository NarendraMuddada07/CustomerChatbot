#from chatbot import chatbot,get_data
from flask import *
from flask_recaptcha import ReCaptcha
#import mysql.connector
import sqlite3
import os
import pandas as pd
from werkzeug.utils import secure_filename
import pandas as pd
import nltk
import random
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key=os.urandom(24)
app.static_folder = 'static'

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin1')  # Adjust encoding if needed
    df.dropna(inplace=True)  # Remove any missing values
    return df

# Train TF-IDF model
def train_model(questions):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)
    return vectorizer, tfidf_matrix

# Find the best match
def get_answer(user_query, vectorizer, tfidf_matrix, questions, answers):
    query_vector = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    best_match_index = similarities.argmax()
    return answers[best_match_index]



# API route to handle user messages
@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.json
    print("data is ",data)
    user_query =user_message = data.get("message", "").strip().lower()  # Normalize input
    print("user_message is ",user_message)
    file_path = "dataset.csv"  # Replace with actual file path
    df = load_data(file_path)
    
    questions = df.iloc[:, 0].tolist()  # First column = questions
    answers = df.iloc[:, 1].tolist()  # Second column = answers
    
    vectorizer, tfidf_matrix = train_model(questions)

    answer = get_answer(user_query, vectorizer, tfidf_matrix, questions, answers)
    print("Predicted Answer:", answer)

    bot_reply = answer
    return jsonify({"response": bot_reply})
    
@app.route("/")
def home():
    return render_template('index.html')


if __name__ == "__main__":
    # app.secret_key=""
    app.run(debug=True) 
