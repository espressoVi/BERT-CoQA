import collections
import glob
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (AdamW, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup)
from processors.coqa import Extract_Features, Processor, Result
from processors.metrics import get_predictions
from transformers import BertModel, BertPreTrainedModel, BertTokenizer, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import pickle

train_file="coqa-train-v1.1.json"
predict_file="coqa-dev-v1.0.json"
pretrained_model="bert-base-uncased"
epochs = 1.0
evaluation_batch_size=16
MIN_FLOAT = -1e30
max_seq_length = 512 

class BertBaseUncasedModel(BertPreTrainedModel):
    def __init__(self,config,activation='relu'):
        super(BertBaseUncasedModel, self).__init__(config)
        self.bert = BertModel(config)
        hidden_size = config.hidden_size

        self.fc = nn.Linear(hidden_size,hidden_size, bias = False)
        self.fc2 = nn.Linear(hidden_size,hidden_size, bias = False)
        self.rationale_modelling = nn.Linear(hidden_size,1, bias = False)
        self.attention_modelling = nn.Linear(hidden_size,1, bias = False)
        self.span_modelling = nn.Linear(hidden_size,2,bias = False)
        self.unk_modelling = nn.Linear(2*hidden_size,1, bias = False)
        self.yes_no_modelling = nn.Linear(2*hidden_size,2, bias = False)
        self.relu = nn.ReLU()

        self.beta = 5.0
        self.init_weights()

    def forward(self,input_ids,segment_ids=None,input_masks=None,start_positions=None,end_positions=None,rationale_mask=None,cls_idx=None, attn = False, block = -1):

        outputs = self.bert(input_ids,token_type_ids=segment_ids,attention_mask=input_masks,
                head_mask = None,output_hidden_states = True, output_attentions = True)
        _, bert_pooled_output,hidden_states, attentions = outputs
        output_vector = hidden_states[block] 
        attentions = list(attentions)

        start_end_logits = self.span_modelling(output_vector)
        start_logits, end_logits = start_end_logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)
        #Rationale modelling 
        rationale_logits = self.relu(self.fc(output_vector))
        rationale_logits = self.rationale_modelling(rationale_logits)
        rationale_logits = torch.sigmoid(rationale_logits)

        output_vector = output_vector * rationale_logits

        attention  = self.relu(self.fc2(output_vector))
        attention  = (self.attention_modelling(attention)).squeeze(-1)
        input_masks = input_masks.type(attention.dtype)
        attention = attention*input_masks + (1-input_masks)*MIN_FLOAT
        attention = F.softmax(attention, dim=-1)
        attentions.append(attention)

        attention_pooled_output = (attention.unsqueeze(-1) * output_vector).sum(dim=-2)
        cls_output = torch.cat((attention_pooled_output,bert_pooled_output),dim = -1)

        rationale_logits = rationale_logits.squeeze(-1)

        unk_logits = self.unk_modelling(cls_output)
        yes_no_logits = self.yes_no_modelling(cls_output)
        yes_logits, no_logits = yes_no_logits.split(1, dim=-1)
        return attentions

def convert_to_list(tensor):
    return tensor.detach().cpu().tolist()

def Write_attentions(model, tokenizer, device, dataset_type = None):
    dataset, examples, features = load_dataset(tokenizer, evaluate=True,dataset_type = dataset_type)
    evalutation_sampler = SequentialSampler(dataset)
    evaluation_dataloader = DataLoader(dataset, sampler=evalutation_sampler, batch_size=evaluation_batch_size)
    res = []
    for c in tqdm(range(1,13),desc = "Block: "):
        su = []
        for batch in evaluation_dataloader:
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0],"segment_ids": batch[1],"input_masks": batch[2],"attn":True,"block":c}
                example_indices = batch[3]
                outputs = model(**inputs)
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                doc_tok = eval_feature.tokens
                attentions = outputs
                attentions = [output[i].detach().cpu().numpy() for output in attentions]
                rational_mask = np.array(eval_feature.rational_mask)
                _ones = np.where(rational_mask == 1)[0]
                try:
                    r_start,r_end = _ones[0],_ones[-1]+1
                except:
                    continue
                su.append(np.sum(attentions[12][r_start:r_end]))
        res.append((np.mean(su),np.std(su)))
    for i in range(12):
        print(f"{i} & {res[i][0]:.6f} & {res[i][1]:.6f}\\\\")

def load_dataset(tokenizer, evaluate=True, dataset_type = None):
    processor = Processor()
    examples = processor.get_examples("data", 2,filename=predict_file, threads=12, dataset_type = dataset_type)
    features, dataset = Extract_Features(examples=examples,
                tokenizer=tokenizer,max_seq_length=512, doc_stride=128, max_query_length=64, is_training=not evaluate, threads=12)
    return dataset, examples, features


def main(model_directory,dataset_type):
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    config = BertConfig.from_pretrained(pretrained_model)
    model = BertBaseUncasedModel.from_pretrained(model_directory)
    tokenizer = BertTokenizer.from_pretrained(model_directory, do_lower_case=True)
    model.to(device)
    for j in dataset_type:
        print(model_directory,j)
        Write_attentions(model, tokenizer, device,dataset_type = j)

if __name__ == "__main__":
    main(model_directory = "Bert_base",dataset_type = [None,'RG','TS'])
    main(model_directory = "Bert_comb2",dataset_type = [None,'RG','TS'])

#def attention_res(attention,head,r_start,r_end,length):
#    assert head < len(attention)
#    if head == -1:
#        attention = np.mean(attention, axis = 0)
#    else:
#        attention = attention[head]
#    assert attention.shape == (max_seq_length,max_seq_length)
#    su,su_r,su_nr = [],[],[]
#    for i in range(length):
#        eta = np.sum(attention[i][r_start:r_end])
#        eta = (eta*length) / (r_end - r_start)
#        if r_start <= i < r_end:
#            su_r.append(eta)
#        else:
#            su_nr.append(eta)
#        su.append(eta)
#    return np.mean(su), np.mean(su_r),np.mean(su_nr)
