
import warnings

import pandas

warnings.filterwarnings('ignore')
data = pandas.read_excel('D:\\PyProject\\LDA_demo\\data\\data.xlsx')
data.head()

import re
import jieba


def chinese_word_cut(mytext):
    jieba.load_userdict('D:\PyProject\Huggingface_Toturials\Lib\site-packages\jieba\dict.txt')  # 这里你可以添加jieba库识别不了的网络新词，避免将一些新词拆开
    jieba.initialize()
    # 文本预处理 ：去除一些无用的字符只提取出中文出来
    new_data = re.findall('[\u4e00-\u9fa5 A-Z a-z 0-9 +#&]+', mytext, re.S)
    new_data = " ".join(new_data)

    # 文本分词
    seg_list_exact = jieba.cut(new_data, cut_all=True)
    result_list = []
    with open('D:\\PyProject\\LDA_demo\\data\\停用词库.txt', encoding='utf-8') as f:  # 可根据需要打开停用词库，然后加上不想显示的词语
        con = f.readlines()
        stop_words = set()
        for i in con:
            i = i.replace("\n", "")  # 去掉读取每一行数据的\n
            stop_words.add(i)

    for word in seg_list_exact:
        if word not in stop_words and len(word) > 1:
            result_list.append(word)
    return " ".join(result_list)


data["content_cutted"] = data.content.apply(chinese_word_cut)
data.head()

from sklearn.feature_extraction.text import CountVectorizer

n_features = 500  # 提取1000个特征词语.
tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df=0.5,
                                min_df=7)
tf = tf_vectorizer.fit_transform(data.content_cutted)



from sklearn.decomposition import LatentDirichletAllocation
n_topics = 100  # 这里是设置LDA分类的主题个数，因为这里我们已经知道了每个内容的标签共有8个类型
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='batch',
                                learning_offset=50,
                                doc_topic_prior=0.1,
                                topic_word_prior=0.01,
                               random_state=666)  # 关于模型的参数,可查看官方文档
lda.fit(tf)


def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx}:" )
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword
# 输出每个主题对应词语
n_top_words = 10
tf_feature_names = tf_vectorizer.get_feature_names_out()
topic_word = print_top_words(lda, tf_feature_names, n_top_words)



import numpy as np
topics=lda.transform(tf)
topics[0] # 查看第一篇文章的主题概率
topic = []
for t in topics:
    topic.append(list(t).index(np.max(t)))
data['topic']=topic
data.to_excel("data_topic.xlsx",index=False)  # 将结果保存为Excel文件

import pyLDAvis
# import pyLDAvis.sklearn
# # import ipython
# # pyLDAvis.enable_notebook()
# pic = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
# pyLDAvis.save_html(pic, 'lda_pass' + str(n_topics) + '.html')  # 将可视化结果打包为html文件
# pyLDAvis.show(pic, local=False)

import matplotlib.pyplot as plt

plexs = []
scores = []
n_max_topics = 16  # 这里值可自行设置
for i in range(1, n_max_topics):
    lda = LatentDirichletAllocation(n_components=i, max_iter=50,
                                    learning_method='batch',
                                    learning_offset=50, random_state=666)
    lda.fit(tf)
    plexs.append(lda.perplexity(tf))
    scores.append(lda.score(tf))

n_t = 15  # 区间最右侧的值。注意：不能大于n_max_topics
x = list(range(1, n_t))
plt.plot(x, plexs)
plt.xlabel("number of topics")
plt.ylabel("perplexity")
plt.show()