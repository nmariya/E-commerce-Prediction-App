from flask import Flask, render_template, jsonify, request
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Text model - RNN with LSTM units
class RNN_lstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_lstm, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.LSTMCell(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input1, hidden):
        combined = torch.cat((input1, hidden), 1)
        hidden = self.i2h(combined)[0]
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# Dictionary
with open('train_dictionary.pk', 'rb') as f:
    res2 = pickle.load(f)
n_words = len(res2)
# Categories dictionary
with open('cat_dictionary.pk', 'rb') as f:
    all_categories = pickle.load(f)
n_categories = len(all_categories)
n_hidden = 128
# Load the saved model states
model = RNN_lstm(n_words, n_hidden, n_categories)
model.load_state_dict(torch.load('char_rnn_lstm_classification_model_state.pt'))

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def main():
    return render_template('index.html')

# Text preprocessing
def clean_doc(doc):
    '''
    This function will clean the text
    '''
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    ps = PorterStemmer()
    tokens = [ps.stem(token) for token in tokens]
    cleaned = (" ".join(tokens))
    return cleaned
def wordToIndex(word):
    '''This function encodes each words from the text to binary. '''
    if word in res2:
        return res2.index(word)
    else:
        return -1  
def lineToTensor(line):
    '''This function converts the text description to tensors to be used in the model'''
    l= line.split()
    tensor = torch.zeros(len(l), 1, n_words)
    for li, word in enumerate(l):
        tensor[li][0][wordToIndex(word)] = 1
    return tensor    

def evaluate_pickl(line_tensor, n_predictions=1):
    '''Prediction is happening here'''
    hidden = model.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    topv_lstm, topi_lstm = output.topk(n_predictions, 1, True)
    predictions_lstm = []
    for i in range(n_predictions):
            value_lstm = topv_lstm[0][i].item()
            category_index_lstm = topi_lstm[0][i].item()
    return all_categories[category_index_lstm]

@app.route('/predict', methods=['GET','POST'])
def predict():
    # Text prediction
    # Sample Input : "Women	Brown	Summer	Casual	Senorita ColorBurst Brown Flats	"
    features=str(request.form.get("gender"))+" "+str(request.form.get("Base Color"))+" "+str(request.form.get("Season"))+" "+str(request.form.get("Usage"))+" "+str(request.form.get("Product Description"))
    t = clean_doc(str(features))
    line_tensor = lineToTensor(t)
    out = evaluate_pickl(line_tensor)
    return render_template('index.html',prediction_text="{}".format(out))

if __name__ == '__main__':
    app.run(port=8080 ,debug=True)
