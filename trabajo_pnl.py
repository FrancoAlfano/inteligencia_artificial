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
from nltk.corpus import stopwords
import re

load_dotenv()

bearer_token = os.environ.get("Bearer")

url ="https://api.twitter.com/2/tweets/search/recent"

params = {
    'query': '#pathofexile OR #poe -is:retweet lang:en',
    'start_time': "2021-10-27T00:00:00Z",
    'end_time': '2021-10-28T00:00:00Z',
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

stop_words = set(stopwords.words('english'))
#new_stopwords = ['poe', 'pathofexile', 'game']
#new_stopwords_list = stop_words.union(new_stopwords)

all_tweets = []
for text in df["tokenized_text"]:
    all_tweets += text
    
cleaned_stop_words = [x for x in all_tweets if not x in stop_words]

text_pos = pos_tag(cleaned_stop_words)
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

# Instanciar Analizador
sentiment_analyzer = SentimentIntensityAnalyzer()
df["negative"] = ""
df["neutral"] = ""
df["positive"] = ""
df["result"] = ""
for index, row in df.iterrows():
    analisis = sentiment_analyzer.polarity_scores(df['tokenized_text'].to_string())
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

fdist = FreqDist(lemmatized)

# Convertir a dataframe
df_fdist = pd.DataFrame.from_dict(fdist, orient='index')
df_fdist.columns = ['Frequency']
df_fdist.index.name = 'Term'
df_fdist.sort_values(by=['Frequency'], inplace=True)

print(df_fdist)

wordcloud = WordCloud(max_words=100, background_color="white").generate(" ".join(lemmatized))
# Mostrar gráfico
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.rcParams['figure.figsize'] = [300, 300]
plt.show()
