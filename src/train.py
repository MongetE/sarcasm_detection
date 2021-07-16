import re
import time
import warnings
import spacy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from collections import Counter
from statistics import mean
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional, Dense, Embedding, Flatten,\
                         Input, LSTM, Dropout, GlobalAveragePooling1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from wordcloud import WordCloud
warnings.filterwarnings('ignore')


def load_training_data(datapath):
    dataframe = pd.read_json(path_or_buf=datapath, dtype={
                                                'is_sarcastic': 'int', 
                                                'headline': 'object',
                                                'article_link': 'object'
                                            })
    dataframe = dataframe.drop(columns='article_link')
    dataframe = dataframe.drop_duplicates().dropna(how='any')

    return dataframe


def load_stopwords(filepath):
    with open('stopwords.txt', 'r', encoding='utf-8') as file: 
        stopwords = file.readlines()
        stopwords = [word.strip() for word in stopwords]

    return stopwords


def process_data(dataframe, stopwords, mode, nlp_pipeline):
    """
        Parameters
        ----------
            dataframe: Pandas Dataframe
                input data to process
            stopwords: list
                list of stopwords to be used
            mode: str
                Kind of processing to apply: 
                    `none`: the data will be passed to the NLP pipeline
                    as is 
                    `stopwords`: stopwords will be removed from the data
                    before being passed to the NLP pipeline
                    `lemmas`: stopwords will be removed from the data
                    before being passed to the NLP pipeline AND tokens
                    will be replaced by their lemmas.
            nlp: Spacy pipeline
                NLP pipeline to apply to the data
    """
    if mode == "none":
        return dataframe

    headline_tmp = list(dataframe.headline)
    start_time = time.time()

    for i in tqdm(range(len(headline_tmp))):
        sentence = headline_tmp[i]
        nlp_sentence = (sentence)

        filtered_lemmas = []
        filtered_tokens = []
        headlines_lemmas = []
        headlines_tokens = []

        unwanted_tokens = ['is', 'are', 'have', 'has', 'had', 'was', 
                           'were', '\'s', 'n\'t', '-', '.', '%']
        
        for token in nlp_sentence:
            if token.pos_ not in ["SYM", "PUNCT", ".", 'DET']:
                if token.text not in stopwords and token.text not in unwanted_tokens:
                    if mode == "stopwords":
                        filtered_tokens.append(token.text)
                    else: 
                        filtered_lemmas.append(token.lemma_)
        
    end_time = (time.time() - start_time)/60
    print(f"Tokenization took {end_time:.2f} minute(s).")
                            
    if len(filtered_lemmas) > 0:
        joined_sentence = ' '.join(word for word in filtered_lemmas)
        joined_sentence = re.sub(r"( 's\b|\w*' s\b)", "'s", joined_sentence)
        joined_sentence = re.sub(r"\s{2,}", " ", joined_sentence)
        headlines_lemmas.append(joined_sentence)
    
        dataframe.headline = headlines_lemmas

        return dataframe


    elif len(filtered_tokens) > 0:
        joined_sentence = ' '.join(word for word in filtered_tokens)
        joined_sentence = re.sub(r"( 's\b|\w*' s\b)", "'s", joined_sentence)
        joined_sentence = re.sub(r"\s{2,}", " ", joined_sentence)
        headlines_tokens.append(joined_sentence)

        dataframe.headline = headlines_tokens

        return dataframe


if __name__ == "__main__":
    datapath =  "input/Sarcasm_Headlines_Dataset_v2.json"








# headline_tmp = list(dataframe.headline)

# headline = []
# for sent in headline_tmp:
#     filtered_list = []
#     for word in sent.split():
#         if word not in stopwords:
#             filtered_list.append(word)
#     join_str = ' '.join([str(ele) for ele in filtered_list])
#     headline.append(join_str)

# sarcasm_dataframe.headline = headline

# max_len = 32
# embedding_dim = 300
# max_words = 30000
# test_samples = 8551
# training_samples = 19952

# tokenizer = Tokenizer(num_words=max_words)
# tokenizer.fit_on_texts(sarcasm_dataframe.headline)
# sequences = tokenizer.texts_to_sequences(sarcasm_dataframe.headline)
# word_index = tokenizer.word_index
# print(f'Found {len(word_index)} unique tokens')

# data = pad_sequences(sequences, maxlen=max_len)
# labels = np.asarray(sarcasm_dataframe.is_sarcastic)
# print(data.shape, labels.shape)

# X_train = data[:training_samples]
# y_train = labels[:training_samples]
# X_test = data[training_samples: training_samples + test_samples]
# y_test = labels[training_samples: training_samples + test_samples]

# embeddings_index = {}

# with open('utils/glove.42B.300d.txt', 'r', encoding='utf-8') as file:
#     glove_embeddings = file.readlines()
    
    
# for line in glove_embeddings:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
    
# print(f'Found {len(embeddings_index)} word vectors')

# embedding_matrix = np.zeros((max_words, embedding_dim))
# for word, i in word_index.items():
#     if i < max_words: 
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None: 
#             embedding_matrix[i] = embedding_vector

# # model = Sequential()
# # model.add(Embedding(max_words, embedding_dim, input_length=max_len))
# # model.add(Flatten())
# # model.add(Dense(32, activation='relu'))
# # model.add(Dense(10, activation='sigmoid'))
# # model.layers[0].set_weights([embedding_matrix])
# # model.layers[0].trainable = False
# # model.summary()
# # model.compile(optimizer='rmsprop', loss='binary_crossentropy', 
# #                 metrics=['accuracy'])
# # history = model.fit(X_train, y_train, epochs=3, batch_size=32, 
# #                     validation_data=(X_test, y_test))

# callback = EarlyStopping(
#     monitor="val_accuracy",
#     patience=5,
#     verbose=1,
#     restore_best_weights=True,
# )
# model = Sequential()
# model.add(Embedding(max_words, embedding_dim, input_length=max_len))
# model.add(Dropout(0.2))
# model.add(Bidirectional(LSTM(64, dropout=0.2, return_sequences=True)))
# model.add(Bidirectional(LSTM(32, dropout=0.2, return_sequences=True)))
# model.add(GlobalAveragePooling1D())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.layers[0].set_weights([embedding_matrix])
# model.layers[0].trainable = False
# model.summary()
# model.compile(loss='binary_crossentropy', optimizer='adam', 
#                 metrics=['accuracy'])
# history = model.fit(X_train, y_train, epochs=15, batch_size=32, 
#                     validation_data=(X_test, y_test), callbacks=[callback])
# model.save("models/sarcasm_bilstm.h5")
# saved_tokenizer = tokenizer.to_json()
# with open(f'models/sarcasm_bilstm_tokenizer.json', 'w', encoding='utf-8') as jsonfile:
#     json.dump(saved_tokenizer, jsonfile, ensure_ascii=False)
                
# # acc = history.history['accuracy']
# # val_acc = history.history['val_accuracy']
# # loss = history.history['loss']
# # val_loss = history.history['val_loss']

# # epochs = range(1, len(acc) + 1)

# # plt.plot(epochs, acc, 'bo', label='Training accuracy')
# # plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# # plt.title('Training and validation accuracy')
# # plt.legend()

# # plt.figure()

# # plt.plot(epochs, loss, 'bo', label='Training loss')
# # plt.plot(epochs, val_loss, 'b', label='Validation loss')
# # plt.title('Training and validation loss')
# # plt.legend()

# # plt.show()