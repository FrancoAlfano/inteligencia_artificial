import os
import re
import requests
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

bearer_token = os.environ.get("Bearer")

url ="https://api.twitter.com/2/tweets/search/recent"

params = {
    'query': '#NLP #MachineLearning -is:retweet',
    'user.fields': 'username',
    'expansions': 'author_id',
    'max_results': 100
}

headers = {
    "Authorization":f"Bearer {bearer_token}",
    "User-Agent":"v2FullArchiveSearchPython"
}


response = requests.get(url, headers=headers, params=params)
df_data = pd.json_normalize(response.json()['data'])
df_users = pd.json_normalize(response.json()['includes']['users'])
df_final = pd.merge(df_data, df_users,how='inner', left_on='author_id', right_on='id').drop([
    'author_id','id_x','name'], axis=1)

df_final.to_csv('tweets_ej2')

if response.status_code != 200:
    raise Exception(response.status_code, response.text)

