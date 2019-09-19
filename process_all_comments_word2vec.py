import pandas as pd
import glove
from gensim.models import Word2Vec

df = pd.read_pickle('/data/nlp_proj/all_comments_per_writer_df.pkl')
print(df.name.nunique())
# df['count']=1
print(df.shape)
print('hiiiiiii')
print(df.name.nunique())

# # """filter out records of names with less than TH_comments unique articles"""
# # TH_comments=90
# # relevant_names=df.groupby('name')['article_id'].nunique()
# # relevant_names.sort_values(ascending=False, inplace=True)
# # relevant_names=relevant_names.index[relevant_names>TH_comments]
# # df=df[df.name.isin(relevant_names)]
# # print(df.shape)
# # print(df.name.nunique())
# #
# #
# #
# # """filter out articles with less than TH_articles commetns"""
# # print('articles before:')
# # print(df.article_id.nunique())
# # TH_articles=50
# # relevant_articles=df.groupby('article_id').size()
# # relevant_articles=relevant_articles.index[relevant_articles>TH_articles]
# # df=df[df.article_id.isin(relevant_articles)]
# # print(df.shape)
# # print('articles after:')
# # print(df.article_id.nunique())
#
# small_df=df[['name','article_id']]
# article_id_u = sorted(df.article_id.unique())
# sentences=[]
# for i,a in enumerate(article_id_u):
#     sentence = small_df[small_df.article_id == a]['name'].str.cat(sep=',').split(',')
#     sentences.append(sentence)
#     if i%500==0:
#         print(i)
#
# #sentences =df[df.article_id==445507]['name'].str.cat(sep=',').split(',')
# model=Word2Vec(sentences=sentences,size=30,window=400,min_count=70, workers=32)
# X=model[model.wv.vocab]
# names=model.wv.vocab.keys()
# names_df=pd.DataFrame(X)
# names_df.index=names
# names_df.to_csv('/data/nlp_proj/all_names_vecs_size_30_min_count_70.csv')
# """
# from sklearn.metrics.pairwise import cosine_similarity
# p=cosine_similarity(X)
# users_df=pd.DataFrame(p,columns=names, index=names)
# users_df.to_csv('word2vec_similarity_all_users_gt_70.csv')
# """
#
#
#
#
# """
# #df.drop_duplicates(subset=['article_id','name'], inplace=True)
# print(df.shape)
#
# from scipy.sparse import csr_matrix
# from itertools import combinations
#
#
# article_id_u = sorted(df.article_id.unique())
# name_u = sorted(df.name.unique())
# res=[]
# for i,a in enumerate(article_id_u):
#     curr_df=small_df[small_df.article_id==a]
#     curr_res=list(combinations(curr_df.name,2))
#     res.extend(curr_res)
#     if i%100==0:
#         print(i)
#
# print(len(res))
# """
# """
# data = df['count'].tolist()
# row = df.name.astype('category', categories=name_u).cat.codes
# col = df.article_id.astype('category', categories=article_id_u).cat.codes
#
# sparse_matrix = csr_matrix((data, (row, col)), shape=(len(name_u), len(article_id_u)))
#
#
#
# from sklearn.metrics.pairwise import cosine_similarity
# p=cosine_similarity(sparse_matrix)
# users_df=pd.DataFrame(p,columns=name_u, index=name_u)
# users_df.to_csv('users_similarity_filtered_names_filtered_first.csv')
# print(users_df.shape)
# """