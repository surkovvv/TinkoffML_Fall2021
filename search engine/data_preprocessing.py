import pandas as pd
import numpy as np
from multiprocessing import Pool, current_process
import re
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
import time

patterns = "[a-zA-Z\d!#$%&'()*+,./:;<=>?@\[\]^_`{|}~—\"\-\n\t]+"
stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()

def lemmatize(doc):
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        if token and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            tokens.append(token)
    #print(current_process())
    if len(tokens) > 2:
        return ' '.join(tokens)
    return None

def preprocess():
    df = pd.read_csv("dataset.csv")
    with Pool(4) as pool:
        texts = pool.map(lemmatize, df['text'].values)
    df["text"] = texts
    df.to_csv("documents.csv", index=False)
    df["title"] = " " + df["title"]
    df["text"] = df["texts"] + df["title"]
    df.to_csv("documents.csv", index=False)

if __name__ == "__main__":
    start = time.time()
    preprocess()
    print("Time to preprocess data : {}".format(time.time() - start))
    #test_df = pd.read_csv("dataset.csv")
    #documents = []
    #with Pool(4) as pool:
    #    documents = pool.map(split, test_df.loc[:10, "text"].values)
    #for i in range(10):
    #    documents.append(test_df.loc[i, "text"].split())
    #print(documents)