S<h1 align="center">Peiyuan's Reading List</h1>



A place to keep track of the NLP papers I read, starting from 2022 Jan.




# Knowledge Point

+ Knowledge Base
1.  An expert system requires structured data. Not just tables with numbers and strings, but pointers to other objects that in turn have additional pointers. The ideal representation for a knowledge base is an object model (often called an ontology in artificial intelligence literature) with classes, subclasses and instances.



# Model and Pretraining


+ [Transformer]() <br>
1. Why scaled dot product: to keep the variance to be asd


+ [BERT]()<br>

+ [A Survey of Transformers]()<br>
1. Three perspectives to improve transformer: a. model effiency: vanilla transformer has quodratic memory and time complexity w.r.t. sequence length. b. model generalization: to introduce inductive bias. c. model adaptation: downstread task.
2. Transformer can be viewed as a GNN defined over a complete directed graph (with
self-loop) where each input is a node in the graph.
3. sparse attention: learned attantion matrix is ofen very sparse.
4. Transformer encoder do not actually need positional encoding
5. Although Post-LN often results in unstable training and divergence, it usually outperforms pre-
LN variants after convergence
6. Universal Transformer (UT): a recurrence-over-depth mechanism

+ [Self-Attention with Relative Position Representations]()<br>
1. clipping distance k: for distrance beyond k, we simply use k as the distrance in relation posiitonal encoding.
2. Insert positional encoding to value and key. 

+ [GPT-1](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) <br>
 1. Left to right LM. Use the last token's final hidden layer to perform classification
 2. Add LM loss during finetuning to speed up convergence and increase performance: Overall, the trend suggests that larger datasets benefit from the auxiliary objective but smaller datasets do not
 3. Use BookCorpus for pretraining. Use cosine schedule. Use BPE. Use GELU
 4. Design a way to convert structured input to flat tokens.
5. Unsupervised pretraining + supervised finetuning + task specific input transformation


+ [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) <br>
 1. Pretrain on WebText, 40GB, use Reddit Karma as indicator.
 2. LayerNorm is moved to the input of each sublayer.
 3. During generation, it ramdomly pick tokens from the top-k result.
 
+ [GPT-3](https://arxiv.org/pdf/2005.14165.pdf) <br>

+ [DeBerta] ()<br>
1. distangled attention
2. enhanced mask decoder
3. vitual adversatiral training

+ [XLNet](https://arxiv.org/pdf/1906.08237.pdf)<br>
1.  BERT limitation: fail to capture depedency between mask (independence assumption between predicted tokens); pretraining-finetuning discrepency
2. Auto Regressive limitation: many downstread tasks need bidirectinal information.
3. Permutation Language Modeling: a. only permute the factorization order (modify attention mask), not sequence order (position encoding remains the same)
4. target position aware: a. Motivation: since it permuate the factorization order, the model do not know the next token position during inference. b. two stream attention: context stream and query stream
5. Training: partial prediction, only predict the last few tokens
5. Ideas from Transformer-XL: relative positional encoding;segment recurrence mechanism
6. Training data: BooksCorpus, English Wiki, Giga5, ClueWeb 2012-B, Common Crawl.
7. Tokenization: SentencePiece.



+ [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf) <br>
0. Motivation: a. Context fragmentation: the fixed-length segments are created by selecting a consecutive chunk of symbols without respecting the sentence or any other semantic boundary. Hence, the model lacks necessary contextual information needed to well predict the first few symbols. b. in traditional mothods, transformer model only generate one token for each forward pass, which is very time comsuming.
1. Segment Level Recurrence with State Reuse: even though gradient does not flow.
2. Relative position encoding: a. inject informaiton into attention score istead of the initial token representation. b. adapt sinusoid formulation. 
3. Use Transformer decoder

+ [RETRO](https://deepmind.com/research/publications/2021/improving-language-models-by-retrieving-from-trillions-of-tokens) <br>
1. Motivation: to explore efficient means of augmenting language models with a massive-scale memory without significantly increasing computations.





+ [Swin Transformer](https://arxiv.org/abs/2103.14030) <br>
1. Promblem of ViT: 1.ViT has quadratic complexity w.r.t. token lengths, making it unsuitable for high resulution set up. 2. ViT is not suitable for dense vision task.
2. Hierarchical Architechure, shifted window.


+ [Universal Transformer]() <br>
0. Motivation: RNNs’ inductive bias towards learning iterative or recursive transformations is important.
1. parallel-in-time recurrent self-attentive sequence model
2. RNN in depth.
3. applying a transition function (shared across position and time) to the outputs of the self-attention mechanism, independently at each position. 
4. Add positional embedding with respect to time and position at each layer.

+ [Performer]()

+ [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks]()
1. To facilitate effective knowledge transfer, however,we often require a large, unlabeled dataset.The teacher model provides the probability logitsand estimated labels for these unannotated samples,and the student network learns from theteacher’s outputs.
2. Use Mean Square Error as error function on the logits of teacher and student network.
3. Design a way to augment the dataset during distillation: maskign, POS swappign, n-gram sampling.
4. The student network is a single layer Bi-LSTM, but it can achieves performance on par with ELMO.

+ [DistillBert]()
1. Distill during pretraining. 40% smaller, 60% faster.
2. Why distillaiton works: some of these "near-zero" probabilities are larger than others and reflect, in part, the generalization capabilities of the model and how well it will perform on the test set3
3. DistilBERT is initialized from the teacher network.


+ [FastBERT]()
1. speed-tunable FastBERT with adaptive inference time
2. Pretrain backbone --> Finetune backbone and teacher classifier --> Bring in student classifier and perform distillation --> Adaptive inference 

+ [Bootstrap your own latent: A new approach to self-supervised learning]()<br>



+ [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language]() <br>

+ [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations]() <br>

+ [BEIT: BERT Pre-Training of Image Transformers]()<br>


# Relation Extraction


+ [Distant Supervision for Relation Extraction with Sentence-Level Attention and Entity Descriptions]()<br>
1. Motivation: a. Multi Instance Learning only choose the instance with highest probability in a mini-batch as the positive instance, which is sub-optimal becuase normally a relation may have many positive instnaces. b. Enity definition may serve as important improtatant for relation extraction.
2. Propose a sentence level attention model which assign different weights to valid and invalid instance in a bag.
+ [Distant Supervision for Relation Extraction with Piecewise Convolutional Neural Networks]()<br>
1. Formulate Distant Supervision as multiple instance learning
2. Use CNN and max pool piecewisely

+ [PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction]() <br>


+ [Few-Shot Named Entity Recognition: A Comprehensive Study]() <br>


+ [Continual Learning for Named Entity Recognition]() <br>
1. Motivation: real world application often needs to add new entity type to NER system, while we can not afford to retrain the whole model.
+ [Template-Based Named Entity Recognition Using BART]()





+ [GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction]()
+ [A Novel Cascade Binary Tagging Framework for Relational Triple Extraction]() <br>
1. Overlapping: a. EntityPairOverlap b.SingleEntityOverlap
2. Motivation: Most existing approach treat relatiosn as discreate label to be assigned,which results in :a. label imbalance, b. confusion caused by EnitityPairOverlap. To takle 
3. Treat relation as a function whose input is subject and object is object.

+ [Revisiting the Negative Data of Distantly Supervised Relation Extraction]() <br>


+ [SENT: Sentence-level Distant Relation Extraction via Negative Training]() <br>

+ [Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling]() <br>
1. Difficulty of Document-level RE: a. Multi Entity: One document commonly contains multiple entity pairs:  it requires the RE model to identify and focus on the part of the document with relevant context for a particular entity pair. b. Multi Label: one entity pair occurs multiple times in the document associated with multiple possible relations.
2. Adaptive Treshholding: Create a new threshholding class to learn the adaptive threshholding during training. Design two loss functions to train ths newly added class.
3. Localized Context Pooling: multiply the last layer attention acore of subject and object to obtain a attention score showing the relavant tokens for both entities, aggregate each token's representation using this score and feed it to classifier. 

# Low Resorce RE

+ [Prototypical Networks for Few-shot Learning]() <br>
1. 
+ [Matching the Blanks: Distributional Similarity for Relation Learning]() <br>
1. Entity Markers: An way to represent the input and output of BERT-like models for RE.
2. Matching the Blanks: a pretraining strategey. Unlike distant supervised learning, MTB treat two sentence with the same entity pairs as the same relation in pretraining. It does not require any human annotation or external Kowledge Base. It only requires a entity extraction tool to extract and link entity. To disable the model from learning superficial clue from entity mention, the mask the entity span with a probability of 0.3
+ [Learning from Context or Names? An Empirical Study on Neural Relation Extraction]() <br>
1. Motivation: which type of information RE models actually grasp to help them extract correct relations: textual context or entity mentions (names): Both context and entity mentions are crucial for RE. As shown in our experiments, while context is the main source to support classification, entity mentions also provide critical information, most of which is the type information of entities
2. Existing RE benchmarks may leak shallow cues via entity mentions, which contribute to the high performance of existing models: we should enhance them via better understanding context and utilizing entity types, while preventing them from simply memorizing entities or exploiting biased cues in mentions.
3. Contribution: investigate an entity-masked contrastive pre-training framework for RE.
4. Unlike MTB, this paper still use Knowledge graph and treat two entity pairs with the same relaiton as the same relation during pretraining, instead if looking at the entity mention. The author argue that this helps the model to learn entity type information.
+ [MapRE: An Effective Semantic Mapping Approach for Low-resource Relation Extraction]() <br>
1. both label-agnostic and label-aware
2. Three stage: pretraining on Wiki, finetuning, tesing.
3. In pretraining stage, three loss: CE betweenn sentence embedding, CE between sentence embedding and relation embedding, mask language modeling.
4. In supervised findtuning, relation embedding * sentence embedding, cross entropy loss.
5. In few shot finetuning, use meta learning (mimic K way N shot). relation embedding * query_sentence_embedding + query_sentence_embedding * support_sentence_embedding.
6. Does 

+ [Exploring Task Difficulty for Few-Shot Relation Extraction]()<br>

+ [FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation]() <>
1. A dataset adapted from Wikipedia. First use distant supervised learning. Then ask crowdworkers to filter out no relation sentence.
+ [Learning Logic Rules for Document-level Relation Extraction]()
+ [A Simple, Strong and Robust Baseline for Distantly Supervised Relation Extraction]()
+ [Separating Retention from Extraction in the Evaluation of End-to-end Relation Extraction]()
+ [A Novel Global Feature-Oriented Relational Triple Extraction Model based on Table Filling]()
+[Gradient Imitation Reinforcement Learning for Low Resource Relation Extraction]()

+ [Modular Self-Supervision for Document-Level Relation Extraction]()
1. Not finished, plz read again
+ [Knowing False Negatives: An Adversarial Training Method for Distantly Supervised Relation Extraction]()
1. STAGE I:Train a binary classifier to classify NA and sentence with relation. Filter out sentences that the model is not very confident about whether it is NA or not as possible False Negative.
2. STAGE II: use bag representation (weighted sum with selective attention). Use adversatiral training (generator and discriminartor) to align the representation of possible false negative with positive examples.
+ [Modeling Relations and Their Mentions without Labeled Text] ()<>
+ [Multi-instance Multi-label Learning for Relation Extraction]()<>
+ [CIL: Con- trastive instance learning framework for distantly su- pervised relation extraction]() <>
# Intent Detection

+ [Few-Shot Intent Detection via Contrastive Pre-Training and Fine-Tuning]() 
1. Contrastive Pretraining: randomly mask 10 tokens to form positive pairs.
2. Fine tuning: contrastive learning where positive piar are from the same class. Soft-max head.
+ [Prototypical Networks for Few-shot Learning]() <>
1. Classics
+ [Dialoglue: A natural language understand- ing benchmark for task-oriented dialogue]() <>
1. Just a benchmark, nothing special .
+ [Discriminative Nearest Neighbor Few-Shot Intent Detection by Transferring Natural Language Inference]()<>
1. Use NLI pretrained model for intent detection. Input 2 sentence  and output binary score. Not worth reading.
2. Backbone: Roberta + NLI
+ [Example-Driven Intent Prediction with Observers]() <>
1. Use observer (like a marker)
2. Use contrastive loss and KNN (like meta learning)
3. Transfer from dataset in different domain also show good result.
4. Backbone: CONVBERT

# Named Entity Recognition
+[Template-Based Named Entity Recognition Using BART]()
1. Among the papers I read, this has got my attention and is of great quality from my perspective. It uses a template-based seq2seq model to perform NER task. Template based in a way that it designs template to convert each entity class in to a natural language. seq2seq in a way that it uses BART as the backbone, with the context sentence as input and template based natural language as output. Given a context sentence and a template based natural language, it calculates the probability for BART to output such a template based netural language. It iterates through all possible class and rank them by probabilty. The class with highest probability will be the final output. So the setup of this paper is quite different from those seq2seq in previous work where BART is to SAMPLE the corresponding labels. 
+ [A Span-Based Model for Joint Overlapped and Discontinuous Named Entity Recognition]()
1. Why people still use graph convulutional networks ?

+ [TriggerNER: Learning with Entity Triggers as Explanations for Named Entity Recognition]()
+[AutoTriggER- Named Entity Recognition with Auxiliary Trigger Extraction]()

+[Continual Learning for Named Entity Recognition]()

+[Data Augmentation for Cross-Domain Named Entity Recognition]()


+[De-biasing Distantly Supervised Named Entity Recognition via Causal Intervention]()


+[Denoising Distantly Supervised Named Entity Recognition via a Hypergeometric Probabilistic Model]()


+[Distantly-Supervised Named Entity Recognition with Noise-Robust Learning and Language Model Augmented Self-Training ]()
+[Few-Shot Named Entity Recognition- A Comprehensive Study]()

+[MECT- Multi-Metadata Embedding based Cross-Transformer for Chinese Named Entity Recognition]()

+[Named Entity Recognition with Small Strongly Labeled and Large Weakly Labeled Data]()



+[Self-training with Noisy Student improves ImageNet classification]()


# Miscellaneous

+ [TextStypeBrush](https://arxiv.org/abs/2106.08385) <br>


+ [SentencePiece](https://arxiv.org/pdf/1808.06226.pdf) <br>
1. SentencePiece, a simple and language independent text tokenizer and detokenizer mainly for Neural Network- based text generation systems.
2. Problem with existing approach of detokenization: it is not reversly convertible. The detokenizer relies on rules to predict white spaces.

+ [WordPiece](https://arxiv.org/pdf/1609.08144.pdf) <br>




# Summary

+ **Second Time Reading Marker** 
+ **First Time Reading Marker**

# Text Similarity
+ [A Survey on Dialogue Systems:Recent Advances and New Frontiers](https://arxiv.org/pdf/1711.01731.pdf) <br>


# Machine Translation
+ [Learning Deep Transformer Mdoels for Machine Translation](https://arxiv.org/pdf/1906.01787.pdf)<br>

+ [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf)<br>


# Computer Vision --  Gait Recognnintion 


+ [GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition]()<br>
1. As a periodic motion, gait can be represented as a single period. Within one period, it was observed that the silhouette in each position has unique appearance. Thus, the author assume the appearance of a silhouette has contained its position information.
2. Input: a set a gait silhouettes --> use CNN to extract frame level features --> Set pooling to obtain set level features --> Horizontal Pyramid Mapping
3. each perid will finally generate a 15872 dimensional representation to represent this person. During testing, Elidean distance will be calculated and the rank 1 result will be the output. 

+ [Gait Lateral Network: Learning Discriminative and Compact Representations for Gait Recognition]() <br>
1. We notice that the silhouettes for different subjects only have subtle differences in many cases, which makes it vital to explore the shallow features encoding the local spatial structural information for gait recognition.
2. learn a compact and discriminative representation for each subject.


+ [Gait Recognition via Effective Global-Local Feature Representation and Local Temporal Aggregation]() <br>
1. Motivation: However, the representations based on global information often neglect the details of the gait frame, while local region based descriptors cannot capture the relations among neighboring regions, thus reduc- ing their discriminativenes
2. Global and Local Feature Extractor(GLFE): composed of multiple Global and Local Convolutional layers (GLConv)
3.  Local Temporal Aggregation (LTA) 
4. Use both Triplet Loss and Cross Entropy Loss
+ [In defense of the triplet loss for person re-identification]() <br>



+ [RealGait: Gait Recognition for Person Re-Identification]() <br>
1. The difference between ReID and Gait recognition
2. Current popular datasets have major drawbacks: they are collected in indoor environment where the subjects are INSTRUCTED to walk instead of walking freely.
3. The author convert an existing ReID into a Gait Recognition dataset by extracting silhouettes.
4. SOTA model on CASIA-B such as GaitGL and GaitSet have poor performance in the wild.

+ [Learning Rich Features for Gait Recognition by Integrating Skeletons and Silhouettes]() <br>

+ [RFMask: A Simple Baseline for Human Silhouette Segmentation with Radio Signals]()<br>

+ [End-to- end model-based gait recognition]()<br>

+ [Multi-Camera Trajectory Forecasting: Pedestrian Trajectory Prediction in a Network of Cameras]()<br>
1. Multi-Camera Trajectory Forecasting
2. Tracking objects (pedestrians) across a large camera net- work requires simultaneously running state-of-the-art algo- rithms for object detection, tracking, and RE-ID.  A successful MCTF model can address this issue by preempting the location of an object-of-interest in a distributed camera network, thereby enabling the system to monitor only selected cameras intelligently. 

+ [ Simple online and realtime tracking with a deep association metric]()<br>


+ [Chained-Tracker: Chaining Paired Attentive Regression Results for End-to-End Joint Multiple-Object Detection and Tracking]()<br>
+ [Mask R-CNN]() <br>
+ [Faster R-CNN]()<br>
+ [Fast R-CNN]()<br>
+ [Masked-attention Mask Transformer for Universal Image Segmentation]()<br>
+[MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation]()<>
+ [Fully convolutional networks for semantic segmentation.]() <brCap@2022>
+[Mask2Former for Video Instance Segmentation]() <br>

+ [Gait Recognition via Disentangled Representation Learning]()<br>
# Computer Vision --  Person Re-Identification
Person ReID basically has the same objective as Gait Recognition. The only difference is that the model input for ReID is a image, while for Gait Recognition it is a video.
+ [Horizontal pyramid matching for person re-identification]()<br>
1. Motivation: missing body part may greatly influennce the model performance for model exploitinng global representation.
2. Use Cross-Entropy Loss.
+ [Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking]
+ [A Discriminatively Learned CNN Embedding for Person Re-identification]() <br>
1. Verification model: take a pair of images as imput and formuate the task as a binary classification. Use triplet loss. Drawbacks: they only use weak re-ID labels, and do not take all the annotated information into consideration. Therefore, the verification network lacks the consideration of the relationship between the image pairs and other images in the dataset.
2. Identification model: formulate the task as multi-class classification. Drawbacks: During testing, the feature is extracted from a fully connected layer and then normalized. The similarity of two images is thus computed by the Euclidean distance between their normalized CNN embed- dings. The major drawback of the identification model is that the training objective is different from the testing procedure, i.e.,it does not account for the similarity measurement between image pairs, which can be problematic during the pedestrian retrieval process.
3. We find that the contrastive loss leads to over-fitting when the number of images is limited.

# Ideas
1. There is a gap between training loss and tesing. Instead, we can always choose a sample as probe during training and calculate cross entropy loss.
GaitGL without crossEntropyLoss 
2. Pretrain out Gait Recognition model of Person ReID dataset. 
3. Contrastive Learning with data augmentation
4. Ramdom Sample a fixed length 的影响
5. Few shot learnin: train a small model to change the last few layer's parameter? 
6. Use existing large scale video dataset such as dataset for activity detection. -- > do human annotation. --> this is only single view
# Reinforcemet Learning 
