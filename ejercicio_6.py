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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import re

load_dotenv()

bearer_token = os.environ.get("Bearer")

url ="https://api.twitter.com/2/tweets/search/recent"

params = {
    'query': '#SpiderMan2 -is:retweet lang:en',
    'tweet.fields': 'created_at',
    'max_results': 50
}

headers = {
    "Authorization":f"Bearer {bearer_token}",
    "User-Agent":"v2FullArchiveSearchPython"
}

rm_urls = r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
rm_hash = r'#'
rm_usr_mention = r'\B\@([\w\-]+)'

response = requests.get(url, headers=headers, params=params)
df = pd.json_normalize(response.json()['data'])

df['text'] = df['text'].str.replace(rm_urls, '', regex=True)
df['text'] = df['text'].str.replace(rm_hash, '', regex=True)
df['text'] = df['text'].str.replace(rm_usr_mention, '', regex=True)
df['text'] = df['text'].str.lower()

tt = TweetTokenizer()

# Aplicar Tokenizer a la columna
tokenized_text = df['text'].apply(tt.tokenize)
df["tokenized_text"] = tokenized_text

print(tokenized_text.to_string())

# Instanciar Analizador
sentiment_analyzer = SentimentIntensityAnalyzer()
# Analizar polaridad de la oración
df["negative"] = ""
df["neutral"] = ""
df["positive"] = ""
df["result"] = ""
for index, row in df.iterrows():
    #Analizar cada review
    analisis = sentiment_analyzer.polarity_scores(tokenized_text.to_string())
    row["negative"] = analisis["neg"]
    row["neutral"] = analisis["neu"]
    row["positive"] = analisis["pos"]
    # Evaluar que valores se considerarán positivo o negativo
    if analisis['compound'] > 0.6 :
        row["result"] = "Positive"
    elif analisis['compound'] <  0.6:
        row["result"] = "Negative"
    else :
        row["result"] = "Neutral"

print(f'Negative= {row["negative"]}')
print(f'Neutral= {row["neutral"]}')
print(f'Positive= {row["positive"]}')
print(f'Result= {row["result"]}')
