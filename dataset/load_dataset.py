import json
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, pipeline
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm

model_path = 'model/distilbert-custom'

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForQuestionAnswering.from_pretrained(model_path)

q = tokenizer('What album made her a worldwide known artist?')
print(q)
