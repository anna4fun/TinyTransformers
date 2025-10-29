# Build a Tiny Transformers (Decoder Only)
This repository contains my code and notes for implementing a tiny decoder only transformers, 
following Andrej Karpathy's YouTube video [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6576s). 
This repo was forked from the lecture's [repo](https://github.com/karpathy/ng-video-lecture)

I plan to integrate MLFlow into this repo for experiment instrumentation.

## Next step:
### Understanding the algorithm
1. [Done] Self attention: is the aggregation aim for predicting the last token? [link to blog](notes/self_attention_what_exactly_is_the_QKV_aggregation_doing.md)
5. [Done] where is the K, Q, V weights training happens? [link to notes](notes/Is_training_of_QKV_happens_with_training_embedding.md)
6. Can I interpret the meaning blobs inside K, Q, V? like CNN
4. what are the different variations of transformers, eg. for large models and small models
2. what is residual path
3. what is the output of transformers?
4. why is a blog marked fig.2 of the attention paper to be:encode as BERT and decode as GPT



### Evaluation during training
1. What are the metrics to evaluate how good the trained GPT is?
2. How to observe these metrics?
3. Can I visualize or test the intermediate output of the GPT?