import nltk
from flask import request
from flask import jsonify
from flask import Flask, render_template
import pickle

#tfidf vectorizer


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
vectorizer=pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    txt=[text]
    txt=vectorizer.transform(txt)
    score=model.predict(txt)

    if(score > 0):
        label = 'This sentence is positive Review'
    elif(score == 0):
        label = 'This sentence is neutral Review'
    else:
        label = 'This sentence is negative Review'

    return(render_template('index.html', variable=label))

if __name__ == "__main__":
    app.run(port='8088', threaded=False, debug=True)

