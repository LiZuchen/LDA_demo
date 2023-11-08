import codecs
import re
import jieba as jb
import gensim
import matplotlib
import numpy as np
from gensim import corpora, models
from matplotlib import pyplot as plt
from pylab import xticks,yticks,np
"""
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
"""
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# 对句子进行分词
def seg_sentence(sentence):
    sentence = re.sub(u'[.]+', u'', sentence)
    jb.add_word('光线摄影学院')		# 这里是加入用户自定义的词来补充jieba词典。
    jb.add_word('曾兰老师')			# 同样，如果你想删除哪个特定的未登录词，就先把它加上然后放进停用词表里。
    jb.add_word('Cplus')
    jb.add_word('自走棋')
    jb.add_word('发布了头条文章')
    jb.add_word('青春有你')
    jb.add_word('青你')
    sentence_seged = jb.cut(sentence.strip())
    stopwords = stopwordslist('D:\\PyProject\\LDA_demo\\data\\stopword.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords and word.__len__()>1:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr
inputs = open('D:\\PyProject\\LDA_demo\\data\\summary_abstract.txt', 'r', encoding='utf-8')
outputs = open('D:\\PyProject\\LDA_demo\\data\\summaryout2.txt', 'w', encoding='utf-8')
for line in inputs:
    line_seg = seg_sentence(line)  # 这里的返回值是字符串
    outputs.write(line_seg + '\n')
outputs.close()
inputs.close()

train = []

fp = codecs.open('D:\\PyProject\\LDA_demo\\data\\summaryout2.txt', 'r', encoding='utf8')
for line in fp:
    if line != '':
        line = line.split()
        train.append([w for w in line])

dictionary = corpora.Dictionary(train)  # 生成字典
corpus = [dictionary.doc2bow(text) for text in train]

num_topics=10
def lda_model_values(num_topics, corpus, dictionary):
    x = []  # x轴
    perplexity_values = []  # 困惑度
    coherence_values = []  # 一致性
    model_list = []  # 存储对应主题数量下的lda模型,便于生成可视化网页

    for topic in range(num_topics):
        print("主题数量：", topic + 1)
        lda_model = models.LdaModel(corpus=corpus, num_topics=topic + 1, id2word=dictionary, chunksize=2000, passes=20,
                                    iterations=20)
        model_list.append(lda_model)
        x.append(topic + 1)
        perplexity_values.append(lda_model.log_perplexity(corpus))

        coherencemodel = models.CoherenceModel(model=lda_model, texts=train, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())
        print("该主题评价完成\n")
    return model_list, x, perplexity_values, coherence_values

model_list, x, perplexity_values, coherence_values = lda_model_values(num_topics, corpus, dictionary)
fig = plt.figure(figsize=(15,5))
plt.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
ax1 = fig.add_subplot(1, 2, 1)
plt.plot(x, perplexity_values, marker="o")
plt.title("主题建模-困惑度")
plt.xlabel('主题数目')
plt.ylabel('困惑度大小')
xticks(np.linspace(1, num_topics, num_topics, endpoint=True)) # 保证x轴刻度为1
ax2 = fig.add_subplot(1, 2, 2)
plt.plot(x, coherence_values, marker="o")
plt.title("主题建模-一致性")
plt.xlabel("主题数目")
plt.ylabel("一致性大小")
xticks(np.linspace(1, num_topics, num_topics, endpoint=True))
plt.show()