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
from flair.data import Sentence
from flair.models import SequenceTagger


def TimeMsg( msg ) :
    logStr = str(datetime.datetime.now()) + ' ' + msg 
    print(logStr)
    return logStr


if __name__ == "__main__":

    # Init Logger : 
    logging.basicConfig(filename='DataAugKeyword.log', level=logging.DEBUG)
    
    # Check file : NameEntity為空的要做Fix的檔案
    fileName = '../Dataset/EN2ZHSUM_train_v3_ne_part3_keyword.csv' # args.testfile
    if fileName == ' ' :
        logging.debug(TimeMsg('Empty of file.'))
        sys.exit()
    
    # Init load file
    trainingPd = pd.read_csv(fileName, sep=',', header=0)
    logging.debug( TimeMsg('Init : Load file success.') )
    
    # Init NameEntityRec Sys
    tagger = SequenceTagger.load("flair/ner-english-large")
    
    output_new_name_entity = []
    
    for index in range(0,len(trainingPd)): 

        if index % 1000 == 0 :
            msg = 'HandlerIndex:{}'.format(index)
            logging.debug( TimeMsg(msg) )
            
        if trainingPd['name_entity'][index] == '[]' :
            tmpSumNes = []
            tmpSentence = Sentence(trainingPd['en_refs'][index])
            tagger.predict(tmpSentence)
            tmpNes = tmpSentence.get_spans('ner')
            for ne in tmpNes : 
                tmpSumNes.append(ne.to_dict()['text'])
            nes_set =[]
            [nes_set.append(i) for i in tmpSumNes if not i in nes_set]
            output_new_name_entity.append(nes_set)
        else :
            output_new_name_entity.append(trainingPd['name_entity'][index])
       
    trainingPd['new_name_entity'] = output_new_name_entity
    trainingPd.to_csv('../Dataset/EN2ZHSUM_train_v3_ne_part3_keyword_emptyfix.csv')
            
            