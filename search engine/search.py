import random
import Cython
from functools import reduce
from multiprocessing import Pool
from gensim.models import Word2Vec
import numpy as np
import math
import pandas as pd
from collections import Counter
import itertools
from scipy import spatial
import time
from data_preprocessing import stopwords_ru, morph

class Document:
    def __init__(self, title, start_text):
        # можете здесь какие-нибудь свои поля подобавлять
        self.title = title
        self.start_text = start_text
    
    def format(self):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.start_text + ' ...']


index = {}
titles = []
w2v_model = Word2Vec.load("w2v_model.model")
"""
def split(text):
    return text.split()

def train_model():
    w2v_model = Word2Vec(
    min_count=10,
    window=5,
    vector_size=300,
    alpha=0.03,
    min_alpha=0.0007,
    sample=6e-5,
    workers=4)

    w2v_model.build_vocab(documents,
                          progress_per=10000)

    w2v_model.train(documents,
                    total_examples=w2v_model.corpus_count,
                    epochs=30,
                    report_delay=1)
    #w2v_model.init_sims(replace=True)
    w2v_model.save("word2vec.model")"""

def build_index():
    # считывает сырые данные и строит индекс
    ratings = []
    global titles
    for i in range(1, 4):
        df = pd.read_csv("part{}.csv".format(i), error_bad_lines=False, engine='python')
        for ind, document in enumerate(df["text"].values):
            for word in set(document.split()):
                if word not in index:
                    index[word] = []
                index[word].append(ind)
            titles.append(df.loc[ind, "title"])
            ratings.append(df.loc[ind, "rating"])
        del df
    for appears in index.values():
        appears.sort(key=lambda index: -ratings[index])  # offline метрика


def count_nDCG(relevant_list):
    DCG = 0
    iDCG = 0
    for i in range(1, len(relevant_list) + 1):
        DCG += relevant_list[i - 1] / math.log2(i + 1)
        iDCG = max(relevant_list[i - 1], iDCG)
    return DCG / iDCG


def map_word_frequency(document):
    return Counter(itertools.chain(*document))


def get_sif_feature_vectors(sentence1, sentence2, word_emb_model=w2v_model):
    sentence1 = [token for token in sentence1.split() if token in word_emb_model.wv.key_to_index]
    sentence2 = [token for token in sentence2.split() if token in word_emb_model.wv.key_to_index]
    word_counts = map_word_frequency((sentence1 + sentence2))
    embedding_size = 300  # size of vector in word embeddings
    a = 0.001
    sentence_set = []
    for sentence in [sentence1, sentence2]:
        vs = np.zeros(embedding_size)
        sentence_length = len(sentence)
        if sentence_length > 0:
            for word in sentence:
                a_value = a / (a + word_counts[word])  # smooth inverse frequency, SIF
                vs = np.add(vs, np.multiply(a_value, word_emb_model.wv[word]))  # vs += sif * word_vector
            vs = np.divide(vs, sentence_length)  # weighted average
        else:
            vs[0] = 1
        sentence_set.append(vs)
    return sentence_set

def get_cosine_similarity(feature_vec_1, feature_vec_2):
    return feature_vec_1.dot(feature_vec_2) / np.linalg.norm(feature_vec_1) / np.linalg.norm(feature_vec_2)


def score(query, document):
    # возвращает какой-то скор для пары запрос-документ, больше -- релевантнее
    vec1, vec2 = get_sif_feature_vectors(query, document.title)
    return get_cosine_similarity(vec1, vec2)


def preprocess_query(query):
    tokens = []
    for token in query.split():
        if token and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]

            tokens.append(token)
    return tokens

def clear_title(title):
    tokens = title.split()
    j = 1
    for i in range(1, len(tokens)):
        j += 1
        if tokens[i][0].isupper(): # начались тэг и хабы
            break
    return ' '.join(tokens[:j])


def retrieve(query):
    """
    Inverted index with handmade intersection with early отбрасывание + offline metrics from dataset
    :param query:
    :return: list of Document
    """
    candidates = []
    for word in set(query.split() + preprocess_query(query)): # хотим получить список id'шников документов, в которых
        # встречаются все слова из запроса
        if word in index:
            candidates.append(index[word])
        # оптимально сделать с помощью n-указателей (по факту тут просто пересечение query.size() упорядоченных сетов)
    if len(candidates) == 0:
        return [Document(title="Sorry, nothing had been found", start_text=":(")]
    intersected_candidates = test_intersect(candidates)
    return [Document(' '.join(titles[ind].split()[:-2]), titles[ind]) for ind in intersected_candidates]


def test_intersect(list_of_lists, topk=500):
    n = len(list_of_lists)
    intersected = []
    everyone_in_range = True
    pointers = [0 for _ in range(n)]
    while everyone_in_range:
        current_max = list_of_lists[0][pointers[0]]
        amount_eq_max = 1
        for ind, pointer in enumerate(pointers[1:]):
            if list_of_lists[ind + 1][pointer] > current_max:
                current_max = list_of_lists[ind + 1][pointer]
                amount_eq_max = 1
            elif list_of_lists[ind + 1][pointer] == current_max:
                amount_eq_max += 1

        if amount_eq_max == n: # т.е. все совпали
            intersected.append(list_of_lists[0][pointers[0]])
            if len(intersected) == topk:
                everyone_in_range = False
                break
            for i in range(n):
                pointers[i] += 1
                if pointers[i] == len(list_of_lists[i]):
                    everyone_in_range = False
                    break
        else:
            for ind, pointer in enumerate(pointers):
                if list_of_lists[ind][pointer] < current_max:
                    pointers[ind] += 1
                    if pointers[ind] == len(list_of_lists[ind]):
                        everyone_in_range = False
                        break

    return intersected

"""
if __name__ == '__main__':

    #build_index()
    #save_obj(index, "index")
    #print("Index building costs {}".format(time.time() - build_time))
    queries = ["яндекс", "python", "программирование на",
               "веб-сервер", "невнятный поиск", "стажировка в",
               "машинное обучение", "глубокое обучение", "Tinkoff", "ODS"]
    for query in queries:
        if len(query.split()) == 1:
            print(index[query])
        retrieve_result = retrieve(query)
        scored = [(doc, score(query, doc)) for doc in retrieve_result]  #
        scored = sorted(scored, key=lambda doc: -doc[1])[:13]
"""