import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

url = r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
hash = r'#'
usr_mention = r'\B\@([\w\-]+)'
df=pd.read_csv("tweets_ej2")

df['text'] = df['text'].str.replace(url, '', regex=True)
df['text'] = df['text'].str.replace(hash, '', regex=True)
df['text'] = df['text'].str.replace(usr_mention, '', regex=True)
df['text'] = df['text'].str.lower()

# Instanciar Tokenizer
tt = TweetTokenizer()
# Aplicar Tokenizer a la columna
tokenized_text = df['text'].apply(tt.tokenize)

df['tokenized_text'] = tokenized_text
tokenized_list = df.explode('tokenized_text')

df.to_csv('tweets_ej4')
# Obtener frecuencia de cada término
fdist = FreqDist(tokenized_list['tokenized_text'])

# Convertir a dataframe
df_fdist = pd.DataFrame.from_dict(fdist, orient='index')
df_fdist.columns = ['Frequency']
df_fdist.index.name = 'Term'
df_fdist.sort_values(by=['Frequency'], inplace=True)

print(df_fdist)

# Generar nube de palabras
wordcloud = WordCloud(max_words=100, background_color="white").generate(df['tokenized_text'].to_string())
# Mostrar gráfico
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.rcParams['figure.figsize'] = [150, 150]
plt.show()

print(df_fdist)
