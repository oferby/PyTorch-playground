import json
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, pipeline
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm

# model_name = 'deepset/bert-base-cased-squad2'
model_name = 'distilbert-base-uncased'
model_path = 'model/distilbert-custom'


def get_set_from_file(filename):
    with open(filename, 'rb') as f:
        return json.load(f)


# print(squad_dict.keys())
# print(json.dumps(squad_dict['data'][-1]['paragraphs'], indent=4, sort_keys=True))


def get_training_set(squad_dict):
    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for p in group['paragraphs']:
            c = p['context']
            for qas in p['qas']:
                q = qas['question']
                if len(qas['answers']) > 0:
                    for answer in qas['answers']:
                        contexts.append(c)
                        questions.append(q)
                        answer['answer_end'] = answer['answer_start'] + len(answer['text']) - 1
                        answers.append(answer)

                else:
                    for answer in qas['plausible_answers']:
                        contexts.append(c)
                        questions.append(q)
                        answer['answer_end'] = answer['answer_start'] + len(answer['text']) - 1
                        answers.append(answer)
    return contexts, questions, answers


squad_dict = get_set_from_file('squad/train-v2.0.json')
train_contexts, train_questions, train_answers = get_training_set(squad_dict)
print(train_contexts[0])
print(train_questions[0])
print(train_answers[0])

# squad_dict = get_set_from_file('squad/dev-v2.0.json')
# val_contexts, val_questions, val_answers = get_training_set(squad_dict)
# print(val_contexts[0])
# print(val_questions[0])
# print(val_answers[0])

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

train_encoding = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
print(train_encoding.keys())
print(train_encoding['input_ids'][0])
print(tokenizer.decode(train_encoding['input_ids'][0]))


# val_encoding = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

def add_token_position(encoding, answers):
    end_positions = []
    start_positions = []
    for i in range(len(answers)):
        start_positions.append(train_encoding.char_to_token(i, answers[i]['answer_start']))
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        end_positions.append(train_encoding.char_to_token(i, answers[i]['answer_end']))
        go_back = 1
        while end_positions[-1] is None:
            end_positions[-1] = train_encoding.char_to_token(i, answers[i]['answer_end'] - go_back)
            go_back += 1
    encoding.update({
        'start_positions': start_positions,
        'end_positions': end_positions
    })


add_token_position(train_encoding, train_answers)
print(train_encoding.keys())


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


train_datasets = SquadDataset(train_encoding)

model = DistilBertForQuestionAnswering.from_pretrained(model_name)
device = torch.device('cuda')
model.to(device)
model.train()
optim = AdamW(model.parameters(), lr=5e-5)
train_loader = DataLoader(train_datasets, batch_size=16, shuffle=True)

for epoch in range(3):
    loop = tqdm(train_loader)
    for batch in loop:
        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                        end_positions=end_positions)
        lose = outputs[0]
        lose.backward()
        optim.step()

        loop.set_description(f'Epoch: {epoch}')
        loop.set_postfix(lose=lose.item())

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
