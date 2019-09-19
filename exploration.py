import pandas as pd
df=pd.read_pickle('/data/nlp_proj/all_comments_per_writer_df2.pkl')
print(df.date.min())
print(df.date.max())
print(df.shape)
