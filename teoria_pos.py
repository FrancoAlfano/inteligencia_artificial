import nltk
example = "The Palace of Westminster serves as the meeting place for both the House of Commons and the House of Lords, the two houses of the Parliament of the United Kingdom. Informally known as the Houses of Parliament after its occupants, the Palace lies on the north bank of the River Thames in the City of Westminster, in central London, England."
# Tokenizar texto
tokenized_text = nltk.word_tokenize(example)
print(tokenized_text)
# Etiquetar texto con pos_tag
text_pos = nltk.pos_tag(tokenized_text)
print(text_pos)