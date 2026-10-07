# Exercise 4.18 NLTK Text Analysis with a Different Text
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

sentence = """
Surabaya is the capital of East Java and the second largest city in Indonesia.
  Politeknik Elektronika Negeri Surabaya was founded in 1988 with support from
  the Japan International Cooperation Agency. Many students in Surabaya study
  Python, machine learning and deep learning to build intelligent applications.
"""
tokens = nltk.word_tokenize(sentence)
print(tokens)
tagged = nltk.pos_tag(tokens)
print(tagged)
entities = nltk.chunk.ne_chunk(tagged)
print(entities)
