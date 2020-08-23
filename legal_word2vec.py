import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#Import dataset
hansard = pd.read_csv("hansard.csv")
hansard_new = hansard[['title', 'motion', 'utt1', 'utt2', 'utt3', 'utt4', 'utt5']]

# Fill in missing values
for col in hansard_new.columns:
    hansard_new[col] = hansard_new[col].fillna(value='Unknown')
print(hansard_new.isnull().sum().sum())

#Join columns
hansard_new = hansard_new.apply(lambda x: " ". join(x), axis=1)

#Text preprocessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

corpus = []

for sentence in hansard_new:
    tokens = word_tokenize(sentence)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = stopwords.words('english')
    tokens = [word for word in tokens if not word in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    corpus.append(tokens)

#Train word2vec on the corpus
import logging
from gensim.models import Word2Vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = Word2Vec(corpus, min_count=5)
model.save("model.bin")

# Test word similarity
w1 = "law"
model.wv.most_similar(positive=w1, topn=5)

w2 = "parliament"
model.wv.most_similar(positive=w2, topn=5)

model.wv.similarity(w1="law", w2="parliament")
model.wv.similarity(w1="government", w2="parliament")

#Fit PCA model to vectors
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

vectors = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

#PCA scatterplot
plt.figure(figsize=(20, 10))
plt.scatter(result[:, 0], result[:, 1], alpha=0.5)
vocabulary = list(model.wv.vocab)
for i, word in enumerate(vocabulary):
    plt.annotate(word, xy=(result[i,0], result[i,1]))
plt.show()
plt.savefig("pca_scatterplot.png")

#KDE density plot
from scipy.stats import gaussian_kde
x = result[:,0]
y = result[:, 1]
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.figure(figsize=(20, 10))
plt.scatter(x, y, c=z, s=50, edgecolor='', alpha=0.5)
words = list(model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i,0], result[i,1]))
plt.show()
plt.savefig("pca_scatterplot_kde.png")
