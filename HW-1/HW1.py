import re
import json

documents = []
def tokenize(text):
    # Remove all types of brackets
    text = re.sub(r'[\[\](){}<>]', '', text)
    
    # Tokenize punctuation as separate tokens unless in a number or word
    text = re.sub(r'(?<=\d)[,.](?=\d)', '', text)
    text = re.sub(r'(?<=\b\w)[,.](?=\w\b)', '', text)
    text = re.sub(r'(?<=\d)[!@#$%^&*+=~`"?|\\/](?=\d)', '', text)
    text = re.sub(r'(?<=\b\w)[!@#$%^&*+=~`"?|\\/](?=\w\b)', '', text)
    
    # Allow contractions and possessives with ≤ 3 letters after [']
    text = re.sub(r"(?<=\b\w)'\b(?=[a-zA-Z]{0,3}\b)", '', text)
    
    # Allow [-] (dash) in the middle of a token
    text = re.sub(r"(?<=\w)-(?=\w)", '', text)
    
    return text
with open("HW1_Appliances_5.json") as file:
    count = 0
    for line in file:
        print("Review number: %d", count)
        review_text = json.loads(line)['reviewText']
        processed_review = re.sub(r'[^\w\s]', '', review_text)
        print(processed_review)
        documents.append(processed_review)
        count += 1
