from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from random import shuffle

app = Flask(__name__)

model = pickle.load(open("new_rf_model.pkl", "rb"))
tfidf = pickle.load(open("new_tfidf.pkl", "rb"))
        
@app.route('/')
def home():
    return render_template('index.html')

def get_review_score(review_text) -> float:
    # todo: to implement the review scoring algorithm
    # scores = [100, 50, 5]

    # shuffle(scores)

    # return float(scores[0])
    print(review_text)
    fe = tfidf.transform([review_text]).toarray()
    result_proba = model.predict_proba(fe)

    print("hasil", result_proba)
    return result_proba[0][1] * 100

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # review = request.form['review']
        # return render_template('index.html', text=review)
        data = request.get_json()
        
        text = data['review'];
        print('review', text)

        result = dict(score = get_review_score(text))
        return jsonify(result)
    
    return render_template('predict.html', text='maaf, gagal')

if __name__ == '__main__':
    app.run(debug=True)