# Surkoff search engine

### Поисковик работает на локальном сервере, реализованном с помощью **Flask**. Поиск осуществляется среди постов с сайта Habr : https://www.kaggle.com/leadness/habr-posts?select=habs.csv(около 2.4 Gb).

## Как выглядит объединенный датасет :
![alt text](https://github.com/surkovvv/TinkoffML_Fall2021/blob/main/search%20engine/dataframe.png)

### Используется модель *Word2Vec* вместе с *SIF(smooth inverse frequency)*, чтобы получить эмбеддинги текстов самих постов(а так же их заголовков и текстов самих запросов). Схожесть определяем с помощью *косинусного расстояния*. Тексты постов предобрабатываются в файлах _data_preprocessing.py_ и в ноутбуке _dataPreprocessing.ipynb_ с помощью *MorphAnalyzer* из *pymorphy2*, используя в том числе *регулярные выражения*. Реализована метрика ранжирования *nDCG*, с помощью которой получилось улучшить качество выдачи (оказалось, что если сравнивать текст поста с текстом запроса, то результат score-функции будет хуже, чем если сравнивать текст заголовка с текстом запроса)


## Пример выдачи запроса "Яндекс"
![alt text](https://github.com/surkovvv/TinkoffML_Fall2021/blob/main/search%20engine/yandex.gif)


## Пример выдачи запроса "Тинькофф"
![alt text](https://github.com/surkovvv/TinkoffML_Fall2021/blob/main/search%20engine/tinkoff.gif)


# Спасибо этим статьям:
### 1. https://medium.com/@adriensieg/text-similarities-da019229c894
### 2. http://proceedings.mlr.press/v37/kusnerb15.pdf

