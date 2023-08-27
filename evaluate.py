# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

# Setting Library
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import pandas as pd
from sklearn.model_selection import train_test_split
from hanspell import spell_checker
from soynlp.normalizer import emoticon_normalize

# koBERT
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

# Transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

device = torch.device('cuda') # GPU 사용

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 7,   # 감정 클래스 수로 조정
                 dr_rate = None,
                 params = None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p = dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict = False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

# 각 데이터가 BERT 모델의 입력으로 들어갈 수 있도록 함수 정의
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):

        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]
    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

# 맞춤법 및 이모티콘 교정 함수
def correct_spelling(sentence) :
    spelled_sent = spell_checker.check(sentence)
    hanspell_sent = spelled_sent.checked
    emoticon_normalized_spell = emoticon_normalize(hanspell_sent, num_repeats = 2)
    return emoticon_normalized_spell

# KoBERT로부터 model, vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model()

# 모델 불러오기
from google.colab import drive
drive.mount('/content/drive')
model = torch.load('/content/drive/MyDrive/toy_project/model.pth')

max_len = 64
batch_size = 64
output = {0 : 'happiness',
          1 : 'neutral',
          2 : 'sadness',
          3 : 'angry',
          4 : 'surprise',
          5 : 'disgust',
          6 : 'fear'}

# 모델 출력 함수
def predict(text) :
  # 맞춤법 및 이모티콘 교정
  text = correct_spelling(text)
  # KoBERT 모델의 입력 데이터 생성
  data = [text, '0']
  dataset = [data]
  # 문장 토큰화
  tokenizer = get_tokenizer()
  tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
  test_data = BERTDataset(dataset,0, 1, tok, max_len, True, False)
  # torch 형식으로 변환
  test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=5)
  model.eval()

  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
      token_ids = token_ids.long().to(device)
      segment_ids = segment_ids.long().to(device)

      valid_length = valid_length
      label = label.long().to(device)

      out = model(token_ids, valid_length, segment_ids)

      test_eval = []
      for i in out: # out = model(token_ids, valid_length, segment_ids)
          logits = i
          logits = logits.detach().cpu().numpy()

      return output[np.argmax(logits)]
