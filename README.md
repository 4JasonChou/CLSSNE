# Conditional Cross-lingual Summarization Generation based on User Specified Named-entities
根據使用者給定之命名實體進行條件式跨語言摘要生成

## 訓練資料集
- NCLS : En2Zh
- NCLS : En2Zh With Named-entities
- 訓練資料位置 LabDevice ( http://192.168.10.205:10529/ )
- 資料位置 : 
	- OrgCorpus : 原始訓練資料集
	- NCLS_En2Zh_Version1 : 包含給定命名實體 & 關注字
	- NCLS_En2Zh_Version2 : 包含給定命名實體 & 實體中文翻譯 & 關注字
	- NCLS_En2Zh_Version3 : 包含給定命名實體 & 關注字 + 段落順序調換

## 各模型介紹
以下模型僅有訓練代碼。
- CLSSNE : 透過指定實體進行控制摘要生成
- CLSSNE+AttentionWord : 指定實體 + 搭配關注字，進行控制摘要生成
- CLSSNE_NKS : 指定實體 + 搭配關注字 + 摘要長度，進行控制摘要生成
- CLSSNE_NKT : 指定實體 + 搭配關注字 + 特殊實體翻譯，進行控制摘要生成
- CLSSNE_Shuffle_And_Final : 透過段落打亂等方式，進行控制摘要生成

## Colab Demo
[Model與CLSSNE_Final-Demo下載網址](https://drive.google.com/drive/folders/1sF9CfQMeaMRj2dLR-DrQ4Wy_T0tX_brU?usp=sharing)

