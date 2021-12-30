# _____________________________________________________________________________________________
# 1. IMPORT NECESSARY LIBRARIES :
import random
import json
import pickle
import numpy as np
import sys
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
# _____________________________________________________________________________________________
# 2. READ CONTENTS OF WORD/CLASS PICKLE FILES CREATED AND USE MODEL BUILT :
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.model')


# _____________________________________________________________________________________________
# 3. FUNCTION TO CLEAN UP SENTENCE :
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# _____________________________________________________________________________________________
# 4. FUNCTION TO BUILD BAG OF WORDS :
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


# _____________________________________________________________________________________________
# 5. FUNCTION TO PREDICT CLASS OF SENTENCE :
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort results by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


# _____________________________________________________________________________________________
# 5. FUNCTION TO GET RESPONSE BY BOT :
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


# _____________________________________________________________________________________________
# 6. APPLICATION :
print("Chat bot is running...")

while True:
    message = input("ENTER A MESSAGE :")
    if message == 'exit':
        sys.exit(0)
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Rohaan-2.O:  " + res)
# _____________________________________________________________________________________________