import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation as lda
from scipy.sparse import csr_matrix

df2 = pd.read_pickle('all_comments_per_writer_df.pkl')
df2['count']=1
print(df2.shape)
print(df2.name.nunique())

article_id_u = sorted(df2.article_id.unique())
name_u = sorted(df2.name.unique())
data = df2['count'].tolist()
row = df2.name.astype('category', categories=name_u).cat.codes
col = df2.article_id.astype('category', categories=article_id_u).cat.codes

sparse_matrix = csr_matrix((data, (row, col)), shape=(len(name_u), len(article_id_u)))
model=lda(n_components=15)
res=model.fit_transform(sparse_matrix)
df1=pd.DataFrame(res)
df1.index=name_u
df1.to_csv('lda_15.csv')