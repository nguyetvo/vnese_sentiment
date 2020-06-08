from flask import Flask, render_template, request, redirect
import os
import sys
import time
import datetime
import math
import re
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
model = tf.keras.models.load_model('./model_my.h5')
words_list = np.load(os.path.join('./data', 'baomoi_word_list.npy'))
words_list = words_list.tolist()
words_vector = np.load(os.path.join('./data', 'baomoi_word_vector.npy'))
words_vector = np.float32(words_vector)
# Loại bỏ các dấu câu, dấu ngoặc, chấm than chấm hỏi, vân vân..., chỉ chừa lại các kí tự chữ và số
word2idx = {w:i for i,w in enumerate(words_list)}
strip_special_chars = re.compile("[^\w0-9 ]+")
def clean_sentences(string):
    #Hàm xử lý, loại bỏ đi những kí tự đặc biệt trả về các từ ở dạng viết thường
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def get_sentence_indices(sentence, max_seq_length, _words_list):
    """
    Hàm này dùng để lấy index cho từng từ trong câu (không có dấu câu, có thể in hoa)
    ----------
    sentence: câu cần xử lý
    max_seq_length: giới hạn số từ tối đa trong câu
    _words_list: bản sao local của words_list, được truyền vào hàm
    ----------
    """
    indices = np.zeros((max_seq_length), dtype='int32')
    
    # Tách câu thành từng tiếng
    words = [word.lower() for word in sentence.split()]
    # Các từ không có trong câu đều không có trong words_list được gán chỉ số của 'unk' 
    unk_idx = word2idx['unk']
    for idx, word in enumerate(words[:max_seq_length]):
      try:
        indices[idx] = word2idx[word]
      except:
        indices[idx] = unk_idx
        
    return indices
def predict_sentence(model, sentence, thres_hold = 0.5):
    indices_sen = get_sentence_indices(sentence, max_seq_length=200, _words_list=words_list)
    ids2vec_sen = tf.nn.embedding_lookup(words_vector, indices_sen)
  # _vec_sen = tf.Session().run(ids2vec_sen)
    _vec_sen=np.array([ids2vec_sen])
    pred = model.predict(_vec_sen)[0][0]

    if 0 < pred <= thres_hold:
        predict = 'NEGATIVE'
    else: 
        predict = 'POSITIVE'
    return [predict,round(pred*100,2)]

@app.route('/') 
def index():
    return render_template('index.html')

@app.route('/sendtext', methods = ['GET','POST'] )
def predict():
    # if request.method == 'POST'
    text = request.form.get('sen')
    predict = predict_sentence(model,text)
    if predict[1] > 50 :
        img = '../static/img/Smiling_Emoji.png'
        pro=predict[1]
    else : 
        img = '../static/img/Sad_Emoji.png'
        pro = 100 - predict[1]

    return render_template('predict.html',pre = predict[0],pro=pro,img = img)
if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000, debug=True)
#   app.run(host='0.0.0.0', port=5000, debug=True)