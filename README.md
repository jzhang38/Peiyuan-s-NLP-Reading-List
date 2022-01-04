<h1 align="center">Peiyuan's Reading List</h1>



A place to keep track of the NLP papers I read.



# Summary
+ [A Survey on Dialogue Systems:Recent Advances and New Frontiers](https://arxiv.org/pdf/1711.01731.pdf) <br>
asdasd



# Model and Pretraining
+ [Character-Aware Neural Language Models](https://arxiv.org/pdf/1508.06615.pdf)：提供一种功能强大，功能强大的语言模型，其可编码子词相关性，同时解决先前模型的罕见字问题，使用更少的参数获得可比较的表现力。 | Yoon et al,2015


# Relation Extraction




# Text Similarity
+ [Siamese Recurrent Architectures for Learning Sentence Similarity](https://scholar.google.com/scholar_url?url=https://ojs.aaai.org/index.php/AAAI/article/view/10350/10209&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=7393466935379636447&ei=KQWzYNL5OYz4yATXqJ6YCg&scisig=AAGBfm0zNEZZez8zh5ZB_iG7UTrwXmhJWg)：Siamese LSTM，一个用来计算句对相似度的模型 | Jonas Mueller et al,2016

+ [Learning Text Similarity with Siamese Recurrent Networks](https://aclanthology.org/W16-1617.pdf)：网络包含4层BiLSTM（64-d hidden），最后一层的BiLSTM的hidden state和cell state进行concat，然后在timestep维度进行average处理，并接一个Dense层（激活函数为tanh），得到的两个Embedding Space进行Cosine sim计算，得到的相似度分数E用于损失函数计算，损失函数使用对比损失函数，计算方法为，损失函数正例：1/4(1-E)^2，负例：E^2(如果E<m)，否则0 | Paul Neculoiu et al,2016

# Machine Translation

# Reinforcemet Learning 
