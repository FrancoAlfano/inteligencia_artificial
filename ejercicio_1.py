import os
import re
import requests
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

bearer_token = os.environ.get("Bearer")

url ="https://api.twitter.com/2/tweets/search/recent"

params = {
    'query': '#NLP @kdnuggets -is:retweet',
    'tweet.fields': 'created_at',
    'max_results': 100
}

headers = {
    "Authorization":f"Bearer {bearer_token}",
    "User-Agent":"v2FullArchiveSearchPython"
}


response = requests.get(url, headers=headers, params=params)
df = pd.json_normalize(response.json()['data'])

df.to_csv('tweets_ej1')

if response.status_code != 200:
    raise Exception(response.status_code, response.text)

