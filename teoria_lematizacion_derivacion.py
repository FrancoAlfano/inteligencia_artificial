
import nltk
from nltk.stem import WordNetLemmatizer
# Importar Lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

love = ["love", "loved", "loving", "loves"]
eat = ["eat", "ate", "eating", "eats", "eaten"]
study = ["study", "studied", "studying", "studies", "student"]
be = ["is", "am", "are", "were", "was"]
car = ["car", "cars", "car's", "cars"]
big = ["big", "bigger", "biggest"]

#A	Adjetivo
#N	Sustantivo
#V	Verbo
#R	Adverbio
for e in eat:
    print(wordnet_lemmatizer.lemmatize(e,"v"))
