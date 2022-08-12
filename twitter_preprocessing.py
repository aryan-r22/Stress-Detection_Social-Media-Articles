
import pandas as pd
import re
import ast
import emoji

path = input("Enter path to scraped data")
addr=glob.glob(path+"/*")

l=[pd.read_csv(x) for x in addr]

df=pd.concat(l, axis=0, ignore_index=True)
#df = df.dropna()

df.columns

df = df.drop(columns=['username', 'acctdesc', 'location', 'following', 'followers',
       'totaltweets', 'usercreatedts', 'tweetcreatedts', 'retweetcount'])

df = df.drop_duplicates(subset=['text'])

df = df.reset_index()

df = df.drop(columns=['index'])

len(df)

def remove_urls(word):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', word)

def remove_symbols(m):
  punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
  for y in punc:
    m=m[len(y):] if m.startswith(y) else m
  m = re.sub('\n+',' ',m)
  if(m=='nan'):
    m=''
  m=emoji.demojize(m)
  m=re.sub(r'^RT[\s]+', '', m)
  m=re.sub(r'#', '', m)
  m=re.sub(' +', ' ', m)
  m=re.sub(r'\@\w+|\#','', m)
  m=re.sub(r':',' ',m)
  m=re.sub(' +', ' ', m)
  return m

from bs4 import BeautifulSoup  
def remove_html(word):
  soup = BeautifulSoup(word, 'lxml')
  html_free = soup.get_text()
  return html_free

text=list(df['text'])
htags=list(df['hashtags'])

text_p=[remove_symbols(remove_html(remove_urls(str(x)))) for x in text]

htags_p=[]
for (i,x) in enumerate(htags):
  if x == '[]':
    htags_p.append([])
  else:
    try:
      x = ast.literal_eval(re.search('({.+})', x).group(0))
    except AttributeError:
      x=x.strip('][').split(', ')
    if isinstance(x, list):
      htags_p.append(x)
    elif isinstance(x, dict):
      htags_p.append([x['text']])
    else:
      temp=[]
      for ss in x:
        temp.append(ss['text'])
      htags_p.append(temp)

assert len(text_p) == len(htags_p)

text_p1=[]
htags_p1=[]

for i in range(len(text_p)):
  if (len(text_p[i])>10):
    text_p1.append(text_p[i])
    htags_p1.append(htags_p[i])

df_new=pd.DataFrame(data = {'text':text_p1,'hashtags':htags_p1})

df_new = df_new.sample(frac=1)



df_new.to_csv(f'preproc_twitter_{len(df_new.index)}.csv')



