# Exercise 4.19 Transformers Sentiment Analysis on Different Sentences
from transformers import pipeline

classifier = pipeline('sentiment-analysis')

sentences = [
    'This is a good movie.',
    'I really love studying artificial intelligence.',
    'The food at this restaurant is delicious.',
    'This is the worst movie I have ever seen.',
    'The lecture was boring and too long.',
    'My laptop keeps crashing, I hate it.',
]

for sentence, result in zip(sentences, classifier(sentences)):
    print("%-50s -> %s (%.4f)" % (sentence, result['label'], result['score']))
