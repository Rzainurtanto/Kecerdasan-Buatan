# Example 4.23 Transformers.py
# https://github.com/huggingface/transformers
# pip install transformers
# https://aka.ms/vs/16/release/vc_redist.x64.exe
from transformers import pipeline

classifier = pipeline('sentiment-analysis')
print(classifier('This is a good movie.'))
