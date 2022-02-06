import sys
import os
import argparse
import pandas as pd
import pickle
import torch
import torch.nn as nn
import logging
import torch.nn.functional as F # 激勵函數
import datetime
import ast
import nltk
import networkx as nx
import numpy as np
from transformers import MBartConfig, MBart50Tokenizer, MBartForConditionalGeneration, AdamW
from transformers import  get_linear_schedule_with_warmup
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from gensim.summarization import keywords
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

def TimeMsg( msg ) :
    logStr = str(datetime.datetime.now()) + ' ' + msg 
    print(logStr)
    return logStr


if __name__ == "__main__":

    # Init Logger : 
    logging.basicConfig(filename='DataAugKeyword.log', level=logging.DEBUG)
    
    # Check file : 預計做資料增量的資料
    fileName = '../Dataset/EN2ZHSUM_test_v3_ne.csv' # args.testfile
    if fileName == ' ' :
        logging.debug(TimeMsg('Empty of file.'))
        sys.exit()
    
    # Init load file
    trainingPd = pd.read_csv(fileName, sep=',', header=0)
    logging.debug( TimeMsg('Init : Load file success.') )
    
    
    output_importKeyword = []
    
    for index in range(0,len(trainingPd)): 
        
        if index % 1000 == 0 :
            msg = 'HandlerIndex:{}'.format(index)
            logging.debug( TimeMsg(msg) )
            
        mContext = str(trainingPd['context'][index])
        # 資料清洗 : 
        if len(mContext) < 128 :
            output_importKeyword.append([])
            continue
            
        mSentences = sent_tokenize(mContext)
        mNameEntitys = ast.literal_eval(trainingPd['name_entity'][index])

        # Step 0 : 資料確認與防呆
        if len(mNameEntitys) == 0 :
            output_importKeyword.append([])
            continue

        # Step 1 : 找出所有包含 NE的 Sentece
        finalSenteces = []
        for sen in mSentences :
            for ne in mNameEntitys :
                tmpRes = sen.find(ne)
                if tmpRes != -1 :
                    finalSenteces.append(sen)
                    break
        
        try :
            # Step 2 : 計算每一句的分數
            # Step 2.1 : N*N個句子的空陣列
            mergeText = ' '.join(finalSenteces)
            tmp = keywords(mergeText).split('\n')
            output_importKeyword.append(tmp)
        except : 
            output_importKeyword.append([])
            msg = 'ErrorIndex:{}'.format(index)
            logging.debug( TimeMsg(msg) )


    trainingPd['ne_keyword'] = output_importKeyword
    trainingPd.to_csv('../Dataset/EN2ZHSUM_test_v3_ne_keyword.csv')
            
            