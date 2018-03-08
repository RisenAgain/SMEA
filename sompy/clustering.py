"""
    First, Thanks to the gread post by Amir Amini[https://www.kaggle.com/amirhamini/d/benhamner/nips-2015-papers/find-similar-papers-knn]
    and brandonrose [http://brandonrose.org/clustering]

    This script describes a method that using word2vec model to cluster nips-2015-papers.
    step one: extract keywords from Title, Abstract and PaperText based on tf-idf
    step two: keywords are used to build the word2vec model
    step three: from keywords to paper document, average the top-n keywords vector to represent the whole paper

    Here are also two clustering method: k-means and Hirerachical clustering.
"""


import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform,pdist
#from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import gensim
import logging
from gensim import models
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

papers_data = pd.read_csv('/home/naveen.pcs16/Experiments/Text_clustering_SMEA/SMEA_Text_auto/data_set/NIPS_2015/Papers.csv')
authors_data = pd.read_csv('/home/naveen.pcs16/Experiments/Text_clustering_SMEA/SMEA_Text_auto/data_set/NIPS_2015/Authors.csv')
authorId_data = pd.read_csv('/home/naveen.pcs16/Experiments/Text_clustering_SMEA/SMEA_Text_auto/data_set/NIPS_2015/PaperAuthors.csv')


"""
    step one:
    extract keywords per paper from Papers Data
    text clean -> tokenize -> stem -> tfidf -> keywords.
"""
def clean_text(text):
    list_of_cleaning_signs = ['\x0c','\n']
    for sign in list_of_cleaning_signs:
        text = text.replace(sign, ' ')
    clean_text = re.sub('[^a-zA-Z]+',' ',text)
    return clean_text.lower()

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def top_tfidf_feats(row, terms, top_n=25):
    top_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [terms[i] for i in top_ids]
    return top_feats

def extract_tfidf_keywords(texts, top_n=25):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, max_features=2000000,
                                      min_df=0.05, stop_words="english",
                                      use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    terms = tfidf_vectorizer.get_feature_names()
    arr = []
    for i in range(0, tfidf_matrix.shape[0]):
        row = np.squeeze(tfidf_matrix[i].toarray())
        feats = top_tfidf_feats(row, terms, top_n)
        arr.append(feats)
    return arr

papers_data['Title_clean'] = papers_data['Title'].apply(lambda x:clean_text(x))
papers_data['Abstract_clean'] = papers_data['Abstract'].apply(lambda x:clean_text(x))
papers_data['PaperText_clean'] = papers_data['PaperText'].apply(lambda x: clean_text(x))



#title2kw = extract_tfidf_keywords(papers_data['Title_clean'],3)
abstract2kw = extract_tfidf_keywords(papers_data['Abstract_clean'], 20)
text2kw = extract_tfidf_keywords(papers_data['PaperText_clean'],100)

"""
    step two:
    word2vec representation
"""
word2vec_model = gensim.models.Word2Vec(abstract2kw+text2kw, size=100, window=5, min_count=5, workers=4)
# model.save("word2vec.model")
# model = Word2Vec.load("word2vec.model")
print "wordvec of robust : ", word2vec_model['robust']

"""
    step three:
    average top-n keywords vectors and compute similarities
"""
doc2vecs = []
for i in range(0, len(abstract2kw)):
    vec = [0 for k in range(100)]
    for j in range(0, len(abstract2kw[i])):
        if abstract2kw[i][j] in word2vec_model:
            vec += word2vec_model[abstract2kw[i][j]]

    for j in range(0, len(text2kw[i])):
        if text2kw[i][j] in word2vec_model:
            vec += word2vec_model[text2kw[i][j]]
    doc2vecs.append(vec)

#similarities = squareform(pdist(doc2vecs, 'cosine'))

"""
    k-means clustering and wordcloud(it can combine topic-models
    to give somewhat more interesting visualizations)
"""
#num_clusters = 13
#km = KMeans(n_clusters=num_clusters)
#km.fit(doc2vecs)
#clusters = km.labels_.tolist()

#print len(doc2vecs)
#from sklearn.metrics import silhouette_score

#ss = silhouette_score(doc2vecs, clusters)
#print "Sil score : ", ss
#papers = { 'Id': papers_data['Id'], 'Title': papers_data['Title'], 'EventType': papers_data['EventType'], 'Cluster': clusters}
#papers_df = pd.DataFrame(papers, index = [clusters] , columns = ['Id', 'Title', 'EventType','Cluster'])
#papers_df['Cluster'].value_counts()

def wordcloud_cluster_byIds(cluId):
    texts = []
    for i in range(0, len(clusters)):
        if clusters[i] == cluId:
            for word in abstract2kw[i]:
                texts.append(word)
            for word in text2kw[i]:
                texts.append(word)

    # wordcloud = WordCloud(max_font_size=40, relative_scaling=.8).generate(' '.join(texts))
    # plt.figure()
    # plt.imshow(wordcloud)
    # plt.axis("off")
    # plt.savefig(str(cluId)+".png")

# wordcloud_cluster_byIds(2)
# wordcloud_cluster_byIds(4)
# wordcloud_cluster_byIds(9)
print doc2vecs[0]
print len(doc2vecs[0])
