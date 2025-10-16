import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = "Sansh2003/roberta-large-merged-subtaskB"

id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
label2id = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}


print("loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=len(label2id), id2label=id2label, label2id=label2id
)
print("done")
print()

def preprocess_single_text(text, tokenizer):
    inputs = tokenizer(text, truncation=True, max_length=512, padding=True, return_tensors="pt")
    return inputs

def get_predictions(input_text: str) -> dict:
    prob_dict = dict.fromkeys(label2id)
    inputs = preprocess_single_text(input_text, tokenizer)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    probs = torch.nn.functional.softmax(logits.squeeze().cpu(), dim=-1) # to get probability
    probs = probs.detach().numpy()    
    for i,k in enumerate(label2id.keys()):
        prob_dict[k] = probs[i]
    prob_dict = {k: float(v) for k,v in sorted(prob_dict.items(), key=lambda item:item[1].item(), reverse=True)}
    print(prob_dict)
    return prob_dict