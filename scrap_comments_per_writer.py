import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np
from multiprocessing import Pool


def return_article_data(article_id):
    try:
        link = 'http://rotter.net/forum/scoops1/' + str(article_id) + '.shtml'
        html = requests.get(link)
        soup = BeautifulSoup(html.content, 'html.parser')

        names, comment_number = zip(
            *[(x.text, x['name']) for x in soup.find_all('a', {"name": re.compile('.*')}) if x.text != ''])
        if names[-1]=="למנהלים:" and comment_number[-1]=="manage":
            names=names[:-1]
            comment_number=comment_number[:-1]

        comment_to = [x.text for x in soup.find_all('a', {'href': re.compile('.*')}) if x.text.startswith("בתגובה להודעה מספר")]
        comment_to = [-1] + [int(re.search('(\d+)$', x).group(0)) for x in comment_to]
        header = soup.find_all('h1')[0].text.split(' ')
        header[-1] = header[-1][:-2]
        header = ' '.join(header)
        comments = [header] + [x.text.split('.', maxsplit=1)[-1][:-2].strip() for x in
                               soup.find_all('font', {'class': 'text16b'})]
        comments_body = [x.text.split('\xa0')[-1].strip().split('\n', maxsplit=1)[0].strip() for x in
                         soup.find_all('font', {'class': 'text15'})]
        maybe_signature = [x.find_all('p')[-1].text.strip() if x.find_all('p')!=[] else '' for x in soup.find_all('font', {'class': 'text15'})]
        #time = [x.find('font', {'color': 'red'}).text for x in
        #        soup.find_all('td', {'align': "LEFT", 'valign': "top", 'width': "50%", 'nowrap': ""})]
        date_time=[x.find('font', {'color':'#000099'}).text for x in soup.find_all('td',{'align':"LEFT", 'valign':"top",'width':"50%", 'nowrap':""})]
        date = [x.split('\xa0')[-1].strip() for x in date_time]
        time = [x.split('\xa0')[-2].strip() for x in date_time]
        assert(all([len(x)==len(names) for x in [comment_number,comment_to,comments,comments_body,maybe_signature,date,time]]))
        df=pd.DataFrame(np.array([names,comment_number,comment_to,comments,comments_body,maybe_signature,date,time]).T, columns=['name','comment_number','comment_to','comments','comments_body','maybe_signature','date','time'])
        df['comment_number']=df['comment_number'].astype(int)
        df['comment_to']=df['comment_to'].astype(int)
        df['original_comment_to']=pd.Series(df.comment_number.where(df.comment_to==0)).shift()[df.comment_to!=0].ffill()
        df['original_comment_to']=df['original_comment_to'].fillna(0)
        df['article_id']=article_id
        df['datetime']=pd.to_datetime(df['date']+' '+df['time'], dayfirst=True)
        df['clean_comments']=df.comments.apply(lambda x: ' '.join(re.findall(r'\w+',x)))
        df['clean_comments_body']=df.comments_body.apply(lambda x: ' '.join(re.findall(r'\w+',x)))
        comment_number2name=dict(zip(df['comment_number'],df['name']))
        df['comment_to_name']=df['comment_to'].map(comment_number2name)
        df['original_comment_to_name']=df['original_comment_to'].map(comment_number2name)
        for col in df.columns:
            if df[col].dtype==object:
                df[col]=df[col].apply(lambda x: np.nan if x==np.nan else str(x).encode('utf-8', 'replace').decode('utf-8'))
        print('success!!')
        return df
    except Exception as ex:
        template = "{0} An exception of type {1} occurred. Arguments:\n{1!r}"
        message = template.format(article_id,type(ex).__name__, ex.args)
        print(message)
pp=return_article_data(574830)
#p = [x for x in range(445501, 523561)]
p = [x for x in range(445501,574832)]

#p = [x for x in range(445501, 445551)]
pool = Pool(processes=10)
res = pool.map(return_article_data, p)
df=pd.concat(res)
print(df.shape)
df.to_csv('/data/nlp_proj/all_comments_per_writer_df2.csv')
df.to_pickle('/data/nlp_proj/all_comments_per_writer_df2.pkl')

