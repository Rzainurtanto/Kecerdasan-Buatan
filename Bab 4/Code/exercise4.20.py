# Exercise 4.20 Transformers Question Answering with a Different Text
from transformers import pipeline

question_answerer = pipeline('question-answering')

context = ('Politeknik Elektronika Negeri Surabaya (PENS) is a vocational higher education '
           'institution located in Surabaya, East Java, Indonesia. It was founded in 1988 with '
           'support from the Japan International Cooperation Agency. PENS offers programs in '
           'electronics, informatics, telecommunications and creative multimedia technology.')

questions = [
    'Where is PENS located?',
    'When was PENS founded?',
    'Who supported the founding of PENS?',
    'What programs does PENS offer?',
]

for question in questions:
    result = question_answerer({'question': question, 'context': context})
    print("Q: %s\nA: %s (score: %.4f)\n" % (question, result['answer'], result['score']))
