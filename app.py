from flask import Flask, render_template, jsonify, request
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
import torch
import torch.nn as nn
from torch.autograd import Variable
# from keras.models import Model, load_model, model_from_json
# import tensorflow as tf
# from keras.initializers import glorot_uniform
# from keras import applications
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.layers import Input, Activation, Dropout, Flatten, Dense, BatchNormalization, Conv2D
# from keras.optimizers import Adam
# from keras import metrics
import numpy as np
# from PIL import Image
# import cv2
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

# # Image model - CNN (InceptionResNetV2)
# def create_model(input_shape, n_out):
    
#     pretrain_model = InceptionResNetV2(
#         include_top=False, 
#         weights='imagenet', 
#         input_shape=input_shape)    
    
#     input_tensor = Input(shape=input_shape)
#     bn = BatchNormalization()(input_tensor)
#     x = pretrain_model(bn)
#     x = Conv2D(128, kernel_size=(1,1), activation='relu')(x)
#     x = Flatten()(x)
#     x = Dropout(0.5)(x)
#     x = Dense(512, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     output = Dense(n_out, activation='sigmoid')(x)
#     model = Model(input_tensor, output)
    
#     return model
# i_model = create_model(input_shape=(299,299,3), n_out=27)

# Dictionary
with open('train_dictionary.pk', 'rb') as f:
    res2 = pickle.load(f)
n_words = len(res2)
# Categories dictionary
with open('./cat_dictionary.pk', 'rb') as f:
    all_categories = pickle.load(f)
n_categories = len(all_categories)
n_hidden = 128
# Load the saved model states
model = RNN_lstm(n_words, n_hidden, n_categories)
model.load_state_dict(torch.load('char_rnn_lstm_classification_model_state.pt'))

# # Load model weights to 
# i_model.load_weights("./ML_models/model.h5")

app = Flask(__name__)

@app.route('/')
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

# # Image preprocessing
# def load_image(path, shape = (299,299,3)):
#     '''This function converts the image to the Image-Net dataset standards'''
#     image = np.array(Image.open(path+'.jpg'))
#     image = cv2.resize(image, (shape[0], shape[1]))
#     image = np.divide(image, 255)
#     return image 

@app.route('/predict', methods=['POST'])
def predict():
    # Text prediction
    test_str = "Women	Brown	Summer	Casual	Senorita ColorBurst Brown Flats	"
    t = clean_doc(test_str)
    line_tensor = lineToTensor(t)
    out = evaluate_pickl(line_tensor)
    return "Prediction from Text Classification model " + out

# @app.route('/i-predict')
# def i_predict():
#     # image prediction
#     image = load_image('images/999999')
#     # imgplot = plt.imshow(image)
#     # plt.show()
#     score_predict = i_model.predict(image[np.newaxis])[0]
#     label_predict = np.argmax(score_predict)
#     return "Prediction from Image Classification model " + all_categories[label_predict]

# @app.route('/predict')
# def evaluate(line_tensor):
#     hidden_lstm = rnn_lstm.initHidden()
#     for i in range(line_tensor.size()[0]):
#         output_lstm, hidden_lstm = rnn_lstm(line_tensor[i], hidden_lstm)
#     return output_lstm
# def predict(input_line, n_predictions=1):
#     with torch.no_grad():
#         output_lstm = evaluate(lineToTensor(input_line))
#     return output_lstm
# INPUT_SHAPE = (299,299,3)
# def image_predict(path, INPUT_SHAPE):
#     image = load_image(path, INPUT_SHAPE)
#     score_predict_t = model.predict(image[np.newaxis])[0]
#     label_predict = np.argmax(score_predict)
#     predicted.append(label_predict)


# @app.route('/fruits')
# def fruits():
#     beers = [
#         {
#             'brand': 'Guinness',
#             'type': 'stout'
#         },
#         {
#             'brand': 'Hop House 13',
#             'type': 'lager'
#         }
#     ]
#     list_of_fruits = ['banana', 'orange', 'apple']
#     list_of_drinks = ['coke', 'milk', beers]
#     return jsonify(Fruits=list_of_fruits, Drinks=list_of_drinks)

if __name__ == '__main__':
    app.run(port=8080 ,debug=True)