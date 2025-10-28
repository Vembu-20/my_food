'''
#translation Using google Translator
from googletrans import Translator

# Initialize the Translator
translator = Translator()

# Input text to translate
paragraph = """Data Science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge from data.
It applies techniques from statistics, machine learning, and computer science to solve real-world problems in various domains."""

translated = translator.translate(paragraph, dest='hi')
print("Original Paragraph:", paragraph)
print("Translated Paragraph:", translated.text)

# using Hugging Face Transformers with the MarianMT model.
!pip install transformers sentencepiece torch

from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-hi'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs, max_length=512)
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translated_text

paragraph = """Data Science is an interdisciplinary field that uses scientific methods, algorithms, and systems to extract knowledge from data. This is the text we are testing for translating paragraph text to hindhi language"""
hindi_translation = translate_text(paragraph)
print("Hindi Translation:", hindi_translation)
'''