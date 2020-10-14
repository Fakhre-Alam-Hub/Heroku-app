from flask import Flask,render_template,url_for,request
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import re
import nltk
import string
from keras.preprocessing.sequence import pad_sequences
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from nltk.corpus import wordnet
from bs4 import BeautifulSoup
from textblob import TextBlob 


model = load_model('sentiment_model.h5')
tokenizer = pickle.load(open('tokenizer.pkl','rb'))
app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def text_preprocess(text):
        
        wn = nltk.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english')

        soup = BeautifulSoup(text, 'html.parser').text
        no_punctuation = "".join([c for c in soup if c not in string.punctuation]).lower()
        no_number  = re.sub(r'[0-9]+', '', no_punctuation)
        tokens = word_tokenize(no_number) 
        no_stopword = [word for word in tokens if word not in stopwords]
        pos = pos_tag(no_stopword)
        lemma = [wn.lemmatize(word[0],get_wordnet_pos(word[1])) for word in pos]
        lemma = ' '.join(lemma)
        
        return lemma
    

    if request.method == 'POST':
        message = request.form['message']
        clean_text = text_preprocess(message)
        blob = TextBlob(clean_text)
        polarity = blob.polarity
        subjectivity = blob.subjectivity
        sentence = np.array([clean_text])
        sentence_sequence = tokenizer.texts_to_sequences(sentence)
        sentence_padded =  pad_sequences(sentence_sequence,maxlen=500,padding='post', truncating='post')
        prediction_prob = model.predict([sentence_padded])[0][0]
        prediction_prob = round(prediction_prob,2)
        if prediction_prob > 0.6:
            my_prediction = 1
        else:
            my_prediction = 0
    return render_template('result.html',prediction = my_prediction,review=message,probability=prediction_prob,polarity=polarity,subjectivity=subjectivity)

if __name__ == '__main__':
	app.run(debug=True)
