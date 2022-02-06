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
import random
from transformers import MBartConfig, MBart50Tokenizer, MBartForConditionalGeneration, AdamW
from transformers import  get_linear_schedule_with_warmup
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
from rouge import Rouge


def TimeMsg( msg ) :
    logStr = str(datetime.datetime.now()) + ' ' + msg 
    print(logStr)
    return logStr

def GetRandomWith20Persent() :
    randomInt = random.randint(1,10)
    if randomInt in [1,2] : 
        return True
    else :
        return False
    
def CsvSummaryHandler( enRefs , zhRefs ) :
    en_List = enRefs.split( '[NEXT]' )
    zh_List = zhRefs.split( '[NEXT]' )
    return en_List[:-1] , zh_List[:-1]

def CsvSummaryZhHandler( zhRefs ) :
    zh_List = zhRefs.split( '[NEXT]' )
    return zh_List[:-1]

def GetRougeScore( expected , predict ) :
    rouge = Rouge()
    rouge_score = rouge.get_scores(expected, predict)
    return ( rouge_score[0]["rouge-1"] , rouge_score[0]["rouge-2"] , rouge_score[0]["rouge-l"] )

def ConvertTokensToString( tokens , speWord ) :
    tmpRes = ''
    for token in tokens :
        tmpRes = tmpRes + token + speWord
    return tmpRes[:-1]

def CalculateRougeScore( tokenizer, expected, predict ) :
    exp = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(expected))
    exp = ConvertTokensToString(tokenizer.convert_ids_to_tokens(exp),' ')
    pre = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(predict))
    pre = ConvertTokensToString(tokenizer.convert_ids_to_tokens(pre),' ')
    return GetRougeScore(exp, pre)

def CreateDataLoader( trainingdata_dict, envBatchSize ) :
    source_ids = trainingdata_dict['input_ids']
    source_mask = trainingdata_dict['attention_mask']
    target_ids = trainingdata_dict['lables']
    all_source_ids = torch.tensor([source_id for source_id in source_ids], dtype=torch.long)
    all_source_masks = torch.tensor([source_mask_one for source_mask_one in source_mask ], dtype=torch.long)
    all_target_ids = torch.tensor([target_id for target_id in target_ids], dtype=torch.long)
    trainningDataSet = TensorDataset( all_source_ids, all_source_masks , all_target_ids )
    trainingDataloader = DataLoader( trainningDataSet ,batch_size=envBatchSize, shuffle=True )
    return trainingDataloader

def CreateTrainingDataLoader( tokenizer , envBatchSize , inputCsvNameList) :

    inputSources_id = []
    inputSources_attentmask = []
    inputlabels_id = []
    
    for inputCsvName in inputCsvNameList :
        inputCsv = pd.read_csv(inputCsvName, sep=',', header=0)       
        for index,rowData in inputCsv.iterrows() :
            # 資料清洗 . 文章長度小於128為異常訓練資料
            if len(str(rowData["context"])) < 50 :
                tmp = 'WarnData:'+str(rowData["context"])
                logging.info( TimeMsg(tmp) )
                continue
                
            tmpContext = str(rowData["context"])
            # Get Random Context with 20% 
            useUpsetContext = GetRandomWith20Persent()
            if useUpsetContext : 
                tmpContext = str(rowData["upset_context"])
            # input - context , 輸入 : 重要NE(s) + [CL] + 內文 ( 10 + 512 )
            tmpContext = NameEntitysHandler(rowData['name_entity']) + '[CL]' + tmpContext
            inputContext = tokenizer(tmpContext, max_length=522, pad_to_max_length=True, return_tensors='pt')
            inputSources_id.append(inputContext['input_ids'].squeeze().tolist())
            inputSources_attentmask.append(inputContext['attention_mask'].squeeze().tolist())
            
            # label - summary , 生成 : 中文摘要
            unitTextTemp_Zh = CsvSummaryZhHandler(rowData['zh_refs'])
            corSumZh =  ('').join(unitTextTemp_Zh)
            multiSum = ZhLabelHandler(tokenizer,corSumZh)
            inputlabels_id.append(multiSum)
            
    print( 'Total TrainingData Size(C):',len(inputSources_id) )
    print( 'Total TrainingData Size(C_mask):',len(inputSources_attentmask) )
    print( 'Total TrainingData Size(S):',len(inputlabels_id) )
    trainingdata_dict = {'input_ids':inputSources_id,'attention_mask':inputSources_attentmask,'lables':inputlabels_id}
    
    return CreateDataLoader( trainingdata_dict, envBatchSize )        

def NameEntitysHandler(neListStr) :
    neHints_tmp = ast.literal_eval(neListStr)
    neHints = ','.join(neHints_tmp)
    return neHints

def ZhLabelHandler( tokenizer, zhSum ):
    zhLabel = []
    with tokenizer.as_target_tokenizer():
        tmpLabel = tokenizer(zhSum, max_length=120,pad_to_max_length=True,return_tensors="pt")
        zhLabel = tmpLabel['input_ids'].squeeze().tolist()
    return zhLabel 

def freeze_params( model ):
    for par in model.parameters():
        par.requires_grad = False

if __name__ == "__main__":
        
    # Env Par :
    trainingDataFileName = ['../Dataset/EN2ZHSUM_train_v3_ne_part1_keyword_emptyfix_upset_28chnage.csv',
                            '../Dataset/EN2ZHSUM_train_v3_ne_part2_keyword_emptyfix_upset_28chnage.csv',
                            '../Dataset/EN2ZHSUM_train_v3_ne_part3_keyword_emptyfix_upset_28chnage.csv']
    testingDataFileName = '../Dataset/EN2ZHSUM_test_v3_ne_keyword_fix.csv' 
    modelName = 'Summary_fzpar_en2zh_v4_ne_input_upset_context_half'
    mEnv_batch_szie = 2 
    mTrainEpoch = 10
    accumulation_step = 8
    tr_loss, logging_loss = 0, 0
    global_step = 0 
    logging_steps = 100  
    
    # Init Logger : 
    logging_file = "mBartFineturn_v28c.log".format(modelName)
    logging.basicConfig(filename=logging_file, level=logging.INFO) 
    logging.info( TimeMsg( "ModelTask:{}, Traingset:{}".format(modelName,trainingDataFileName) ) )
    logging.info( TimeMsg( "Info: Predict important name-entity and summary. with 2:8 UpsetContext" ) )
    logging.info( TimeMsg( "Epoch:{}, BatchSize:{}x{}".format(mTrainEpoch,mEnv_batch_szie,accumulation_step) ) )   

    # Init mBartModel
    mBartModel = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    mBartTokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="zh_CN")
    logging.info( TimeMsg('Init : Model, Tokenizer success.') )

    # Init SpecialToken 
    special_tokens_dict = { 'additional_special_tokens':[] }
    special_tokens_dict['additional_special_tokens'].append('[CL]')
    num_added_toks = mBartTokenizer.add_special_tokens(special_tokens_dict)
    mBartModel.resize_token_embeddings(len(mBartTokenizer))
    mBartTokenizer.save_pretrained("tokenizer_vCommon/".format(modelName))

    # Init load gpu
    #os.environ["CUDA_VISIBLE_DEVICES"] = '1' # Set gpu
    mDevice = torch.device("cuda")
    mBartModel.to(mDevice)
    logging.info( TimeMsg('Init : Load model to Gpu success.') )

    # 優化器設定
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in mBartModel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in mBartModel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    Learning_rate = 2e-5       # 學習率 [Learning_rate = 2e-5]
    optimizer = AdamW(optimizer_grouped_parameters, lr=Learning_rate, eps=1e-8) 

    # 創建訓練資料 : MaxLen : 1024 ( OOM ) , 只能使用 Len : 522  ( Ne + CL + Context )
    trainingDataLoader = CreateTrainingDataLoader(mBartTokenizer,mEnv_batch_szie, trainingDataFileName)
    testingPd = pd.read_csv(testingDataFileName, sep=',', header=0)
    logging.info( TimeMsg('Init : Creat traingData success.') )
    
    # 動態調整學習率
    warmup_steps = 0
    t_total = len(trainingDataLoader) // accumulation_step * mTrainEpoch  # 需要全部資料筆數才能計算 
    scheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total )

    # 參數凍結
    freeze_params(mBartModel.model.shared)
    for d in [mBartModel.model.encoder, mBartModel.model.decoder]:
        freeze_params(d.embed_positions)
        freeze_params(d.embed_tokens)
    logging.info( TimeMsg('Init : Freeze params success.') )
    
    # Fine-Turn
    logging.info( TimeMsg('Running : Fine-turn.') )
    for epoch in range(mTrainEpoch):
        # Train
        logging.info( TimeMsg('Training...') )
        mBartModel.train()
        for index, batch_dict in enumerate(trainingDataLoader):
            batch_dict = tuple(t.to(mDevice) for t in batch_dict)
            outputs = mBartModel( input_ids = batch_dict[0], attention_mask = batch_dict[1], labels = batch_dict[2] )
            loss = outputs[0]
            loss = loss / accumulation_step
            loss.backward()
            tr_loss += loss.item()
            if (index % accumulation_step) == 0:
                optimizer.step() # update
                scheduler.step()
                optimizer.zero_grad() # reset
                global_step += 1
        train_loss = tr_loss / global_step
        tmpStr = 'Epoch:'  + str(epoch) + ' Loss:' + str(train_loss)
        logging.info( TimeMsg(tmpStr) )

        # Save Checkpoint
        logging.info( TimeMsg('Save checkpoint...') )
        mBartModel.save_pretrained("Checkpoint_v28c/point_{}".format(str(epoch)))

        # Eval
        logging.info( TimeMsg('Evaling...') )
        mBartModel.eval()
        rouge1ScoreTotal = 0
        rouge2ScoreTotal = 0
        rougelScoreTotal = 0
        totalDataCount = 0
        errorCount = 0
        for index , row in testingPd.iterrows() :
            tmpContext = NameEntitysHandler(row['name_entity_fix']) + '[CL]' + row['context']
            tmplabelZh = CsvSummaryZhHandler(row['zh_refs'])
            tmpInput = mBartTokenizer( tmpContext, max_length=522, return_tensors='pt' )
            tmpInput.to(mDevice)
            tmpOutput = mBartModel.generate( input_ids = tmpInput['input_ids'],
                                            attention_mask = tmpInput['attention_mask'],
                                            num_beams=4, max_length=120, repetition_penalty=1.2, early_stopping=True)
            tmpOutputStr = [ mBartTokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in tmpOutput ][0]
            
            if index == 0 : 
                logging.info( TimeMsg( str(tmpOutputStr) ) )
            # 假如計算RougeScore出錯 不要影響訓練
            try :
                labelZhSum = ('').join(tmplabelZh)
                r1 , r2 , rl = CalculateRougeScore(mBartTokenizer,labelZhSum,tmpOutputStr)
                rouge1ScoreTotal = rouge1ScoreTotal + r1['f']
                rouge2ScoreTotal = rouge2ScoreTotal + r2['f']
                rougelScoreTotal = rougelScoreTotal + rl['f']
                totalDataCount += 1
            except:
                errorCount += 1
        logging.info( TimeMsg('Score - Rouge1 : '+ str(rouge1ScoreTotal/totalDataCount) ) )
        logging.info( TimeMsg('Score - Rouge2 : '+ str(rouge2ScoreTotal/totalDataCount) ) )
        logging.info( TimeMsg('Score - RougeL : '+ str(rougelScoreTotal/totalDataCount) ) )
        
    # Save Model
    logging.info( TimeMsg('Finish : Fine-turn.') )
