import os
import re
import requests
import pandas as pd
from dotenv import load_dotenv
from nltk import pos_tag
from nltk.tokenize import TweetTokenizer
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

load_dotenv()

bearer_token = os.environ.get("Bearer")

url ="https://api.twitter.com/2/tweets/search/recent"

params = {
    'query': '#pathofexile #poe #scourge -is:retweet lang:en',
    'tweet.fields': 'created_at',
    'max_results': 100
}

headers = {
    "Authorization":f"Bearer {bearer_token}",
    "User-Agent":"v2FullArchiveSearchPython"
}

rm_urls = r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
rm_hash = r'#'
#rm_usr_mention = r'\B\@([\w\-]+)'

response = requests.get(url, headers=headers, params=params)
df = pd.json_normalize(response.json()['data'])

df['text'] = df['text'].str.replace(rm_urls, '', regex=True)
df['text'] = df['text'].str.replace(rm_hash, '', regex=True)
#df['text'] = df['text'].str.replace(rm_usr_mention, '', regex=True)
df['text'] = df['text'].str.lower()

tt = TweetTokenizer()

# Aplicar Tokenizer a la columna
tokenized_text = df['text'].apply(tt.tokenize)
df["tokenized_text"] = tokenized_text

all_tweets = []
for text in df["tokenized_text"]:
    all_tweets += text

text_pos = pos_tag(all_tweets)
#N* -> N
#J* -> A
#V* -> V
#R* -> R
taged_OK = []
for i in range(len(text_pos)):
    if text_pos[i][1][0] == "N":
        text_pos[i] = (text_pos[i][0], "N")
        taged_OK.append(text_pos[i])
    elif text_pos[i][1][0] == "J":
        text_pos[i] = (text_pos[i][0], "A")
        taged_OK.append(text_pos[i])
    elif text_pos[i][1][0] == "V":
        text_pos[i] = (text_pos[i][0], "V")
        taged_OK.append(text_pos[i])
    elif text_pos[i][1][0] == "R":
        text_pos[i] = (text_pos[i][0], "R")
        taged_OK.append(text_pos[i])
taged_OK

wordnet_lemmatizer = WordNetLemmatizer()
lemmatized = []
for word, simbol in taged_OK:
    lemmatized.append(wordnet_lemmatizer.lemmatize(word, simbol.lower()))
lemmatized

wordcloud = WordCloud(max_words=100, background_color="white").generate(" ".join(lemmatized))
# Mostrar gráfico
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.rcParams['figure.figsize'] = [300, 300]
plt.show()
