import re
import warnings
import spacy
import sys
import config
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
    with open(filepath, 'r', encoding='utf-8') as file: 
        stopwords = file.readlines()
        stopwords = [word.strip() for word in stopwords]

    return stopwords


def process_data(dataframe, stopwords, nlp, mode='none'):
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
                Default to none
            nlp: Spacy pipeline
                NLP pipeline to apply to the data
    """
    if mode == "none":
        return dataframe

    headline_tmp = list(dataframe.headline)
    
    headlines_lemmas = []
    headlines_tokens = []

    unwanted_tokens = ['is', 'are', 'have', 'has', 'had', 'was', 
                           'were', '\'s', 'n\'t', '-', '.', '%']

    print("TOKENIZATION IN PROGRESS\n")
    for i in tqdm(range(len(headline_tmp))):
        sentence = headline_tmp[i]
        nlp_sentence = nlp(sentence)

        filtered_lemmas = []
        filtered_tokens = []

        for token in nlp_sentence:
            if token.pos_ not in ["SYM", "PUNCT", ".", 'DET']:
                if token.text not in stopwords and token.text not in unwanted_tokens:
                    if mode == "stopwords":
                        filtered_tokens.append(token.text)
                    else: 
                        filtered_lemmas.append(token.lemma_)

        if mode == "stopwords":
            joined_sentence = ' '.join(word for word in filtered_tokens)
            joined_sentence = re.sub(r"( 's\b|\w*' s\b)", "'s", joined_sentence)
            joined_sentence = re.sub(r"\s{2,}", " ", joined_sentence)
            headlines_tokens.append(joined_sentence)

        else: 
            joined_sentence = ' '.join(word for word in filtered_lemmas)
            joined_sentence = re.sub(r"( 's\b|\w*' s\b)", "'s", joined_sentence)
            joined_sentence = re.sub(r"\s{2,}", " ", joined_sentence)
            headlines_lemmas.append(joined_sentence)
    print("Done")

    if len(headlines_lemmas) > 0:
        dataframe.headline = headlines_lemmas

        return dataframe

    elif len(headlines_tokens) > 0:
        dataframe.headline = headlines_tokens

        return dataframe

def build_model(max_words, max_len, embedding_dimension, name='Bi-LSTM', 
                embedding_matrix=None):
    model = Sequential(name=name)
    model.add(Embedding(max_words, embedding_dimension, input_length=max_len))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64, dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(32, dropout=0.2, return_sequences=True)))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    if embedding_matrix is not None:
        model.layers[0].set_weights([embedding_matrix])
        model.layers[0].trainable = False
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                    metrics=['accuracy'])
    
    return model


def load_glove_embeddings(embeddings_path, max_words, embedding_dimension, word_index): 
    embeddings_index = {}

    with open(embeddings_path, 'r', encoding='utf-8') as file:
        glove_embeddings = file.readlines()

    print("NOW GOING THROUGH GLOVE EMBEDDINGS")
    for i in tqdm(range(len(glove_embeddings))):
        line = glove_embeddings[i]
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    print(f'Found {len(embeddings_index)} word vectors')

    embedding_matrix = np.zeros((max_words, embedding_dimension))
    for word, i in word_index.items():
        if i < max_words: 
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
    print('Done\n')

    return embedding_matrix


def prepare_input_data(dataframe, max_len, max_words):
    tokenizer = Tokenizer(num_words=max_words, 
                        filters='!"#$%&()*+,-./:;<=>?@[\\]^\'_`{|}~\t\n')
    tokenizer.fit_on_texts(dataframe['headline'])
    word_index = tokenizer.word_index
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)  
    X = tokenizer.texts_to_sequences(dataframe['headline'])
    X = pad_sequences(X, maxlen=max_len)
    y = dataframe['is_sarcastic']

    return word_index, X, y


def run(glove): 
    """
        Run the training process with or without glove embeddings. 
        Use glove embeddings by default.
    """

    max_len = config.MAXLEN
    embedding_dimension = config.EMBEDDINGDIM
    max_words = config.MAXWORDS
    epochs = config.EPOCHS
    batch_size = config.BATCH_SIZE

    callback = EarlyStopping(
                                monitor="val_accuracy",
                                patience=5,
                                verbose=1,
                                restore_best_weights=True,
                            )
     
    datapath =  config.TRAINING_FILE
    sarcasm_dataframe = load_training_data(datapath)
    stopwords = load_stopwords(config.STOPWORDS)
    nlp = spacy.load('en_core_web_sm', exclude=["ner" 'textcat', 'parser', 
                                                'textcat_multilabel', 'transformer', 
                                                'senter', 'sentencizer'])
    processed_dataframe = process_data(sarcasm_dataframe, stopwords, 
                                       nlp, mode='lemmas')
    word_index, X, y = prepare_input_data(processed_dataframe, max_len, max_words)

    if glove:
        embedding_matrix = load_glove_embeddings(config.GLOVE, max_words, 
                                                embedding_dimension, word_index)
    else: 
        embedding_matrix = None 

    model = build_model(max_words, max_len, embedding_dimension, 
                        embedding_matrix=embedding_matrix)
    print('Switching to training)')
    model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=batch_size, 
             callbacks=[callback])



if __name__ == "__main__":
    try: 
        glove = sys.argv[1]
        if glove == "False": 
            glove = False
    except IndexError:
        glove = True

    run(glove)