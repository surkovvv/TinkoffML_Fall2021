# Surkoff search engine

### Поисковик работает на локальном сервере, реализованном с помощью **Flask**. Поиск осуществляется среди постов с сайта Habr : https://www.kaggle.com/leadness/habr-posts?select=habs.csv(около 2.4 Gb).

## Как выглядит объединенный датасет :
![alt text](https://github.com/surkovvv/TinkoffML_Fall2021/blob/main/search%20engine/dataframe.png)

### Используется модель Word2Vec вместе с SIF(smooth inverse frequency), чтобы получить эмбеддинги текстов самих постов, которые предобрабатываются в файлах _data_preprocessing.py_ и в ноутбуке _dataPreprocessing.ipynb_ с помощью **MorphAnalyzer** из **pymorphy2**, используя в том числе **регулярные выражения**. 


## Пример выдачи запроса "Яндекс"
![alt text](https://github.com/surkovvv/TinkoffML_Fall2021/blob/main/search%20engine/yandex.gif)


## Пример выдачи запроса "Тинькофф"
![alt text](https://github.com/surkovvv/TinkoffML_Fall2021/blob/main/search%20engine/tinkoff.gif)
