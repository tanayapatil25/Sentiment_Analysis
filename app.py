from flask import Flask, render_template, request
import pickle
from text_preprocessor import TextPreprocessor  # Make sure you have your text_preprocessor module

def create_app():
    app = Flask(__name__, static_url_path='/static')

    # Load the sentiment analysis model
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        if request.method == 'POST':
            review = request.form['review']
            preprocessed_review = TextPreprocessor().transform([review])
            prediction = model.predict(preprocessed_review)[0]
            return render_template('index.html', prediction=prediction)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)











