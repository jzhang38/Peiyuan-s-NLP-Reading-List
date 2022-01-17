<h1 align="center">Peiyuan's Reading List</h1>



A place to keep track of the NLP papers I read.







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
+ [Self-Attention with Relative Position Representations] <br>
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
4. target position aware: a. Motivation: since it permuate the faxtorization order, the model do not know the next token position during inference. b. two stream attention: context stream and query stream
5. Training: partial prediction, only predict the last few tokens
5. Ideas from Transformer-XL: relative positional encoding;segment recurrence mechanism
6. Training data: BooksCorpus, English Wiki, Giga5, ClueWeb 2012-B, Common Crawl.
7. Tokenization: SentencePiece.



+ [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf) <br>
0. Motivation: a. Context fragmentation: the fixed-length segments are created by selecting a consecutive chunk of symbols without respecting the sentence or any other semantic boundary. Hence, the model lacks necessary contextual information needed to well predict the first few symbols. b. in traditional mothods, transformer model only generate one token for each forward pass, which is very time comsuming.
1. Segment Level Recurrence with State Reuse: even thought gradient does not flow.
2. Relative position encoding: a. inject informaiton into attention score istead of the initial token representation. b. adapt sinusoid formulation. 
3. Use Transformer decoder

+ [RETRO](https://deepmind.com/research/publications/2021/improving-language-models-by-retrieving-from-trillions-of-tokens) <br>
1. Motivation: to explore efficient means of augmenting language models with a massive-scale memory without significantly increasing computations.

+ [WebGPT](https://arxiv.org/pdf/2112.09332.pdf) <br>



+ [Swin Transformer](https://arxiv.org/abs/2103.14030) <br>
1. Promblem of ViT: 1.ViT has quadratic complexity w.r.t. token lengths, making it unsuitable for high resulution set up. 2. ViT is not suitable for dense vision task.
2. Hierarchical Architechure, shifted window.
+ [TextStyleBrush]() <br>


+ [The Natural Language Decathlon:Multitask Learning as Question Answering](https://arxiv.org/pdf/1806.08730.pdf) <br>


+ [ERNIE-Doc: A Retrospective Long-Document Modeling Transformer]()

+ [Universal Transformer]()

+ [Performer]()

+ [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks]()

+ [DistillBert]()
+ [FastBERT]()


# Relation Extraction



+ [PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction]() <br>


+ [Few-Shot Named Entity Recognition: A Comprehensive Study]() <br>


+ [Continual Learning for Named Entity Recognition]() <br>

+ [Template-Based Named Entity Recognition Using BART]()

+ [How Knowledge Graph and Attention Help]() <br>


+ [Potential Relation and Global Correspondence Based Joint Relational Triple Extraction - Yefeng Zheng]() <br>

+ [GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction]()
+ [A Novel Cascade Binary Tagging Framework forRelational Triple Extraction]() <br>
1. Overlapping: a. EntityPairOverlap b.SingleEntityOverlap
2. Motivation: Most existing approach treat relatiosn as discreate label to be assigned,which results in :a. label imbalance, b. confusion caused by EnitityPairOverlap. To takle 
3. Treat relation as a function whose input is subject and object is object.
+ [Revisiting the Negative Data of Distantly Supervised Relation Extraction]() <br>


+ [SENT: Sentence-level Distant Relation Extraction via Negative Training]() <br>

+ [Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Poolin]() <br>


+ [Place Hodler]() <br>


+ [Place Hodler]() <br>


+ [Place Hodler]() <br>


# Miscellaneous

+ [TextStypeBrush](https://arxiv.org/abs/2106.08385) <br>


+ [SentencePiece](https://arxiv.org/pdf/1808.06226.pdf) <br>
1. SentencePiece, a simple and language independent text tokenizer and detokenizer mainly for Neural Network- based text generation systems.
2. Problem with existing approach of detokenization: it is not reversly convertible. The detokenizer relies on rules to predict white spaces.

+ [WordPiece](https://arxiv.org/pdf/1609.08144.pdf) <br>




# Summary

+ **Second Time Reading Marker** 
+ **First Time Reading Marker**

+ [Place Hodler]() <br>


+ [Place Hodler]() <br>


+ [Place Hodler]() <br>


+ [Place Hodler]() <br>


+ [Place Hodler]() <br>


+ [Place Hodler]() <br>


+ [Place Hodler]() <br>

# Text Similarity
+ [A Survey on Dialogue Systems:Recent Advances and New Frontiers](https://arxiv.org/pdf/1711.01731.pdf) <br>


# Machine Translation
+ [Learning Deep Transformer Mdoels for Machine Translation](https://arxiv.org/pdf/1906.01787.pdf)<br>

+ [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf)<br>

# Reinforcemet Learning 
