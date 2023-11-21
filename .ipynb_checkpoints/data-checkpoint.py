# import hashlib
# import re
# import sys
# import tarfile
# from collections import Counter, defaultdict
# from pathlib import Path

# import matplotlib.pyplot as plt
# import requests
# from IPython.display import Image

# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from wordcloud import WordCloud
# nltk.download('all')

# # Read train, val, and test sets into string objects
# train_data = Path('wikitext-103/wiki.train.tokens').read_text()
# val_data = Path('wikitext-103/wiki.valid.tokens').read_text()
# test_data = Path('wikitext-103/wiki.test.tokens').read_text()


# # Store regular expression pattern to search for wikipedia article headings 기사 제목양식
# heading_pattern = '( \n \n = [^=]*[^=] = \n \n )'


# # Split out train headings and articles 기사제목, 내용 분리
# train_split = re.split(heading_pattern, train_data)
# train_headings = [x[7:-7] for x in train_split[1::2]]
# train_articles = [x for x in train_split[2::2]]

# # Split out validation headings and articles
# val_split = re.split(heading_pattern, val_data)
# val_headings = [x[7:-7] for x in val_split[1::2]]
# val_articles = [x for x in val_split[2::2]]


# # Split out test headings and articles
# test_split = re.split(heading_pattern, test_data)
# test_headings = [x[7:-7] for x in test_split[1::2]]
# test_articles = [x for x in test_split[2::2]]


# # Remove casing, punctuation, special characters, and stop words and also lemmatize the words on a subset of the first 110 articles in the train data
# my_new_text = re.sub('[^ a-zA-Z0-9]|unk', '', train_data[:2010011])
# stop_words = set(stopwords.words('english'))
# lemma = WordNetLemmatizer()
# word_tokens = word_tokenize(my_new_text.lower())
# filtered_sentence = (w for w in word_tokens if w not in stop_words)
# normalized = " ".join(lemma.lemmatize(word) for word in filtered_sentence)


