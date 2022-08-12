import pandas as pd
import re
import emoji
import glob


path = input("Enter path to scraped data")
addr=glob.glob(path+"/*")

l=[pd.read_csv(x) for x in addr]

df=pd.concat(l, axis=0, ignore_index=True)
#df = df.dropna()

df.columns

df = df.drop_duplicates(subset=['title','body'])

df = df.drop(columns=['score','id','url','comms_num'])

df = df.reset_index()

len(df)

df=df.drop(columns='index')

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
  m=re.sub(r'\@\w+|\#','', m)
  m=re.sub(r':',' ',m)
  m=re.sub(' +', ' ', m)
  return m

from bs4 import BeautifulSoup  
def remove_html(word):
  soup = BeautifulSoup(word, 'lxml')
  html_free = soup.get_text()
  return html_free

titles=list(df['title'])
body=list(df['body'])

title_p=[remove_symbols(remove_html(remove_urls(str(x)))) for x in titles]
body_p=[remove_symbols(remove_html(remove_urls(str(x)))) for x in body]

assert len(title_p) == len(body_p)

title_p1=[]
body_p1=[]

print(f"Max length of title is {max([len(x) for x in title_p])}")
print(f"Min length of title is {min([len(x) for x in title_p])}")
print(f"Max length of body is {max([len(x) for x in body_p])}")
print(f"Min length of body is {min([len(x) for x in body_p])}")

for i in range(len(title_p)):
  if (~(len(title_p[i])<10 and len(body_p[i])<10) and len(title_p)>10):
    title_p1.append(title_p[i])
    body_p1.append(body_p[i])

for i in range(len(title_p)):
  if (len(title_p[i])>10 and len(body_p[i])>10):
    title_p1.append(title_p[i])
    body_p1.append(body_p[i])

k=['\[removed\]','[\n']
for i,xx in enumerate(body_p): ##Change to body_p1
  if (xx==k[0] or xx==k[1]):
    body_p1[i]=''

df_new=pd.DataFrame(data = {'title':title_p1,'body':body_p1})  ##Change to p1

df_new = df_new.sample(frac=1)



df_new.to_csv(f'preproc_reddit_{len(df_new.index)}.csv')


