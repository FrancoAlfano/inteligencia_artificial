import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example = "The Palace of Westminster serves as the meeting place for both the House of Commons and the House of Lords, the two houses of the Parliament of the United Kingdom. Informally known as the Houses of Parliament after its occupants, the Palace lies on the north bank of the River Thames in the City of Westminster, in central London, England."
# General lista de stop words 
stop_words = set(stopwords.words('english'))

# Aplicar Tokenizer
tokenized_text = word_tokenize(example)
#print(tokenized_text) 

example_no_stopwords = [x for x in tokenized_text if not x.lower() in stop_words]
#print(example_no_stopwords)