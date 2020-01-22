import pandas as pd
import numpy as np
import json, re
from tqdm import tqdm
from uuid import uuid4

## Torch Modules
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

## PyTorch Transformer
from transformers import RobertaModel, RobertaTokenizer
from transformers import RobertaForSequenceClassification, RobertaConfig

df = pd.read_csv('../data/myPersonality/mypersonality_final.csv', encoding='latin1')
print("The size of data is {0}".format(df.shape[0]))
labels = ['cEXT','cNEU','cAGR','cCON','cOPN']

class Intents(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe
        
    def __getitem__(self, index):
        utterance = self.data.text[index]
        label = self.data.label[index]
        X, _  = prepare_features(utterance)
        y = label_to_ix[self.data.label[index]]
        return X, y
    
    def __len__(self):
        return self.len

def prepare_features(seq_1, max_seq_length = 300, 
             zero_pad = False, include_CLS_token = True, include_SEP_token = True):
    ## Tokenzine Input
    tokens_a = tokenizer.tokenize(seq_1)

    ## Truncate
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    ## Initialize Tokens
    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)
    ## Add Tokens and separators
    for token in tokens_a:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    ## Input Mask 
    input_mask = [1] * len(input_ids)
    ## Zero-pad sequence lenght
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
    return torch.tensor(input_ids).unsqueeze(0), input_mask

for label2 in labels:
    print("Classify for label", label2)
    data = df[['STATUS', label2]]
    data = data.rename(columns={'STATUS': "text", label2: "label"})
    label_to_ix = {}
    for label in data.label:
        for word in label.split():
            if word not in label_to_ix:
                label_to_ix[word]=len(label_to_ix)
    config = RobertaConfig.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification(config)

    train_size = 0.8
    train_dataset=data.sample(frac=train_size,random_state=200).reset_index(drop=True)
    test_dataset=data.drop(train_dataset.index).reset_index(drop=True)

    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = Intents(train_dataset)
    testing_set = Intents(test_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.cuda()

    # Parameters
    params = {'batch_size': 1,
          'shuffle': True,
          'drop_last': False,
          'num_workers': 1}

    training_loader = DataLoader(training_set, **params)
    testing_loader = DataLoader(testing_set, **params)

    loss_function = nn.CrossEntropyLoss()
    learning_rate = 1e-05
    optimizer = optim.Adam(params =  model.parameters(), lr=learning_rate)

    max_epochs = 3

    last_accuracy = 0
    last_loss = 0

    model = model.train()
    for epoch in tqdm(range(max_epochs)):
        print("EPOCH -- {}".format(epoch))
        for i, (sent, label) in enumerate(training_loader):
            optimizer.zero_grad()
            sent = sent.squeeze(0)
            if torch.cuda.is_available():
                sent = sent.cuda()
                label = label.cuda()
            output = model.forward(sent)[0]
            _, predicted = torch.max(output, 1)
            
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            
            if i%100 == 0:
                correct = 0
                total = 0
                for sent, label in testing_loader:
                    sent = sent.squeeze(0)
                    if torch.cuda.is_available():
                        sent = sent.cuda()
                        label = label.cuda()
                    output = model.forward(sent)[0]
                    _, predicted = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (predicted.cpu() == label.cpu()).sum()
                accuracy = 100.00 * correct.numpy() / total
                print('Iteration: {}. Loss: {}. Accuracy: {}%'.format(i, loss.item(), accuracy))
    correct = 0
    total = 0
    for sent, label in testing_loader:
        sent = sent.squeeze(0)
        if torch.cuda.is_available():
            sent = sent.cuda()
            label = label.cuda()
        output = model.forward(sent)[0]
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted.cpu() == label.cpu()).sum()
    accuracy = 100.00 * correct.numpy() / total
    last_accuracy = accuracy
    last_loss = loss.item()
    print('Iteration: {}. Loss: {}. Accuracy: {}%'.format(i, loss.item(), accuracy))
    print(last_accuracy, last_loss)
    torch.save(model.state_dict(), 'models_output/'+label2+'_'+str(last_accuracy)+'_'+str(last_loss)+'_'+ str(uuid4())+'.pth')
