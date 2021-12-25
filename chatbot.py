import random
import json
import pickle
import numpy as np
import os

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras import models

from tensorflow.keras.models import load_model


def chatbot_initialise():
    lemmatizer = WordNetLemmatizer()

    filepath = os.path.join(os.getcwd(), 'test_data')
    with open(os.path.join(filepath, 'intents.json'), 'r') as f:
        intents = json.loads(f.read())

    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)

    with open('classes.pkl', 'rb') as g:
        classes = pickle.load(g)

    model = load_model('chatbotmodel.h5')

    return lemmatizer, intents, words, classes, model


def clean_up_sentence(lemmatizer, sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(lemmatizer, words, sentence):
    sentence_words = clean_up_sentence(lemmatizer, sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def predict_class(lemmatizer, model, sentence, classes):
    bow = bag_of_words(lemmatizer, sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list


def get_response(message):
    lemmatizer, intents_json, words, classes, model = chatbot_initialise()

    intents_list = predict_class(lemmatizer, model, message, classes)

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result
