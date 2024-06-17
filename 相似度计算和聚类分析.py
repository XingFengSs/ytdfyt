from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
# 加载已经训练好的模型
model = Word2Vec.load("word2vec.model")


# 计算词向量之间的相似度
similarity = model.wv.similarity('端午', '假期')
print("Similarity between '端午' and '假期':", similarity)

# 获取所有词的词向量
all_words = list(model.wv.key_to_index.keys())
word_vectors = np.array([model.wv[word] for word in all_words])


# 使用K-means对词向量进行聚类
num_clusters = 5
kmeans_model = KMeans(n_clusters=num_clusters)
kmeans_model.fit(word_vectors)

# 获取每个词所属的簇
labels = kmeans_model.labels_

# 将词和其所属的簇打印出来
word_cluster = zip(all_words, labels)
for word, cluster in word_cluster:
    print(f"Word: {word}, Cluster: {cluster}")

