import codecs
import re
import jieba as jb
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import logging
import gensim
from gensim import corpora, models
import matplotlib.pyplot as plt
import matplotlib
from pylab import xticks,yticks,np

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

from gensim import corpora

import pyLDAvis.gensim
# d=pyLDAvis.gensim.prepare(lda, corpus, dictionary)
# pyLDAvis.show(d)		#展示在浏览器
# # pyLDAvis.displace(d) #展示在notebook的output cell中
# d=pyLDAvis.gensim.prepare(lda, corpus, dictionary)
# pyLDAvis.save_html(d, 'lda_pass10.html')	# 将结果保存为该html文件
'''
lda: 计算好的话题模型
corpus: 文档词频矩阵
dictionary: 词语空间
浅蓝色的表示这个词在整个文档中出现的频率（权重），
深红色的表示这个词在这个主题中所占的权重。
右上角可以调节一个参数λ，其作用接着往下看。
'''
# plex=[]
# scores=[]
# for i in range(2,10):
#     print("n_topics: ",i)
#     lda=LatentDirichletAllocation(n_components=i,max_iter=50,learning_method='batch',learning_offset=50,random_state=0)
#     lda.fit(corpus)
#     plex.append(lda.perplexity(corpus))
#     scores.append(lda.score(corpus))
# import matplotlib.pyplot as plt
# x=list(range(2,10))
# plt.plot(x,plex)
# plt.show()

"""
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
"""


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

        coherencemodel = models.CoherenceModel(model=lda_model, texts=train,dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())
        print("该主题评价完成\n")
    return model_list, x, perplexity_values, coherence_values


if __name__ == '__main__':
# 调用准备函数
    inputs = open('D:\\PyProject\\LDA_demo\\data\\summary_abstract.txt', 'r', encoding='utf-8')

    outputs = open('D:\\PyProject\\LDA_demo\\data\\summaryout2.txt', 'w',encoding='utf-8')
    for line in inputs:
        line_seg = seg_sentence(line)  # 这里的返回值是字符串
        outputs.write(line_seg + '\n')
    outputs.close()
    inputs.close()

    train = []

    fp = codecs.open('D:\\PyProject\\LDA_demo\\data\\summaryout2.txt','r',encoding='utf8')
    for line in fp:
        if line != '':
            line = line.split()
            train.append([w for w in line])

    dictionary = corpora.Dictionary(train)#生成字典
    # dictionary.save('qzone.dict')  # 把字典存储下来，可以在以后直接导入

    #字典处理完以后，使用gensim提供的API就可以直接给出语料库的向量化表现形式了
    corpus = [dictionary.doc2bow(text) for text in train]
    # corpora.MmCorpus.serialize('corpus_bow.mm', corpus)  # 存储语料库
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # num_topics：主题数目
    # passes：训练伦次
    # num_words：每个主题下输出的term的数目
    num_topics0 = 8
    chunksize = 2000
    passes = 20
    eval_every = None
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token
    # model = LdaModel(
    #     corpus=corpus,
    #     id2word=id2word,
    #     chunksize=chunksize,
    #     alpha='auto',
    #     eta='auto',
    #     iterations=iterations,
    #     num_topics=num_topics,
    #     passes=passes,
    #     eval_every=eval_every
    # )
    lda = LdaModel(corpus=corpus, id2word=dictionary,num_topics=num_topics0, passes=passes)
    x=lda.print_topics(num_words = 10)
    for topic in x:
        termNumber = topic[0]
        print(topic[0], ':', sep='')
        print(topic[1])
        # listOfTerms = topic[1].split('+')
        # for term in listOfTerms:
        #     listItems = term.split('*')
        #     print('  ', listItems[1], '(', listItems[0], ')', sep='')
    d=pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(d, 'lda_pass10.html')
    num_topics = 10
    model_list, x, perplexity_values, coherence_values = lda_model_values(num_topics, corpus, dictionary)

# 绘制困惑度和一致性折线图
    fig = plt.figure(figsize=(15,5))
    plt.rcParams['font.sans-serif']=['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(x, perplexity_values, marker="o")
    plt.title("主题建模-困惑度")
    plt.xlabel('主题数目')
    plt.ylabel('困惑度大小')
    xticks(np.linspace(1, num_topics, num_topics, endpoint=True))
# 保证x轴刻度为1
    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(x, coherence_values, marker="o")
    plt.title("主题建模-一致性")
    plt.xlabel("主题数目")
    plt.ylabel("一致性大小")
    xticks(np.linspace(1, num_topics, num_topics, endpoint=True))
    plt.savefig('D:\\PyProject\\LDA_demo\\data\\out'+(str)(num_topics0)+'.png')
    # plt.show()


    pyLDAvis.show(d)		#展示在浏览器
