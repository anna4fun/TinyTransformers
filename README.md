# Build a Tiny Transformers (GPT2)
This repository contains my code and notes for implementing a tiny decoder only transformers, 
following Andrej Karpathy's YouTube video [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6576s). 
This repo was forked from the lecture's [repo](https://github.com/karpathy/ng-video-lecture)

## Blogs:
### Attention is All You Need readout
1. [Done] Self attention: is the aggregation aim for predicting the last token? [link to blog](notes/self_attention_what_exactly_is_the_QKV_aggregation_doing.md)
5. [Done] where is the K, Q, V weights training happens? [link to notes](notes/Is_training_of_QKV_happens_with_training_embedding.md)
6. Can I interpret the meaning blobs inside K, Q, V? like CNN
4. what are the different variations of transformers, eg. for large models and small models
2. what is residual path
3. What is the output of transformers? A prediction(aka, generation) of next token given a sentence.
4. Why is a blog marked fig.2 of the attention paper to be:encode as BERT and decode as GPT. 
   Short Answer: The Attention is All You Need paper Figure 2 contains an encoder and decoder connected by a cross-attention sharing of Q and K. However, none of the BERT and GPT models implemented this encoder+decoder architecture: BERT is an Encoder that is trained to look both directions of the sequence and make any position prediction, while GPT is a Decoder because it's products ChatGPT is a human chat format which naturally goes left to right sequence understanding and generation.


### Evaluation during training
![swanlab_dashboard_overfit_one_batch.png](pictures/swanlab_dashboard_overfit_one_batch.png)
1. What are the metrics to evaluate how good the trained GPT is?
2. How to observe these metrics?
3. Can I visualize or test the intermediate output of the GPT?

### More
1. Now that I know how to write the core code of GPT2, how to use different models on HuggingFace? Start with the GPT family (GP2, GPT3, GPT-oss)
2. 