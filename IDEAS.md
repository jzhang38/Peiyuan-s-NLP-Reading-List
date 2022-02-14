# Ideas
1. There is a gap between training loss and tesing. Instead, we can always choose a sample as probe during training and calculate cross entropy loss.







GaitGL without crossEntropyLoss 

2. Pretrain our Gait Recognition model of Person ReID dataset. 
3. Contrastive Learning with data augmentation
4. Ramdom Sample a fixed length 
5. Few shot learnin: train a small model to change the last few layer's parameter? 
6. Use existing large scale video dataset such as dataset for activity detection. -- > do human annotation. --> this is only single view









1. a. The reason why type embedding does not work is because of it is not pretrained this way. b. However, type embedding has eplicit advantage, which means it can directly model the subject span and object span. 
    c. Can we add type embedding in to later laye instead?
2. Contrastive learning, seq2seq, change order?
3. NER: change enetity mention ,contrastive learning?
3. BART + few resource? (zero-shot?)
4. how to denoise distantly supervised training: assume the majority of the training sample is correct. so instance whose vector representation is far from others will be treated as noise data.
5. Data augmetation in relation extraction: mask subj & obj span. sample from pretrained language's prediciton ==> data augmentationn & contrastive learning.  ==> use it for distant relation extraction
# Suggestions