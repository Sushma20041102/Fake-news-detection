from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ✅ Function to preprocess new user input
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return " ".join(text)

@app.route("/")
def home():
    return render_template("index.html")  # Create an HTML form for input

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form["news_text"]
    cleaned_text = clean_text(news_text)
    transformed_text = vectorizer.transform([cleaned_text])  # Convert text
    prediction = model.predict(transformed_text)[0]
    
    result = "Real News 📰" if prediction == 1 else "Fake News 🚨"
    
    return render_template("index.html", prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)
