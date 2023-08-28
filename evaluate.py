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
from bert_utils import BERTClassifier, BERTDataset

# koBERT
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

# Transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

device = torch.device('cuda') # GPU 사용

# 맞춤법 및 이모티콘 교정 함수
def correct_spelling(sentence) :
    spelled_sent = spell_checker.check(sentence)
    hanspell_sent = spelled_sent.checked
    emoticon_normalized_spell = emoticon_normalize(hanspell_sent, num_repeats = 2)
    return emoticon_normalized_spell

# KoBERT로부터 model, vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model()

# 모델 경로 지정
model_path = '/content/drive/MyDrive/toy_ai/model.pth'
model = torch.load(model_path)

max_len = 64
batch_size = 64
output = {0 : '행복',
          1 : '보통',
          2 : '슬픔',
          3 : '분노',
          4 : '놀람',
          5 : '불쾌함',
          6 : '두려움'}

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