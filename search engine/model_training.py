import pandas as pd
import numpy as np
from gensim.models import Word2Vec

def train_model(file, model):
    data = [text.split() for text in pd.read_csv(file, index_col="Unnamed: 0")["text"].values]

    if model.corpus_count == 0:
        model.build_vocab(data, progress_per=10000)
    else:
        model.build_vocab(data, update=True)

    print(model.corpus_count)
    model.train(data, total_examples=model.corpus_count, epochs=30)
    # w2v_model.init_sims(replace=True)
    #w2v_model.save("word2vec.model")
    return model

if __name__ == "__main__":
    w2v_model = Word2Vec(
        min_count=10,
        window=5,
        vector_size=300,
        alpha=0.03,
        min_alpha=0.0007,
        sample=6e-5,
        workers=4)
    w2v_model = train_model("part1.csv", w2v_model)
    print("First part done")
    w2v_model = train_model("part2.csv", w2v_model)
    print("Second part done")
    w2v_model = train_model("part3.csv", w2v_model)
    print("Third part done")
    w2v_model.save("w2v_model.model")