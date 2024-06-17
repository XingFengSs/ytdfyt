from gensim.models import Word2Vec


# 第一步：读取预处理文件并创建句子列表
sentences = []
with open('processed_chinese_corpus.txt', 'r', encoding='utf-8') as file:
    for line in file:
        sentences.append(line.strip().split())

# 打印前 10 行句子，检查是否正确读取
for sentence in sentences[:10]:
    print(' '.join(sentence))

# 第二步：使用句子列表训练 Word2Vec 模型
# 设置模型参数
vector_size = 100  # 向量维度
window = 5         # 上下文窗口大小
min_count = 1      # 最小词频
workers = 4        # 训练使用的CPU线程数

# 创建并训练模型
model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

# 第三步：保存训练好的模型
model.save("word2vec.model")

# 如果需要加载模型，可以使用以下代码
loaded_model = Word2Vec.load("word2vec.model")
