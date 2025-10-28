# Build a Tiny Transformers (Decoder Only)
This repository contains my code and notes for implementing a tiny decoder only transformers, 
following Andrej Karpathy's YouTube video [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6576s). 
This repo was forked from the lecture's [repo](https://github.com/karpathy/ng-video-lecture)

I plan to integrate MLFlow into this repo for experiment instrumentation.

next step:
1. attention: is the aggregation works for predicting the last token?
2. what are the different variations of transformers, eg. for large models and small models
2. what is residual path
3. what is the output of transformers?
4. why is a blog marked fig.2 of the attention paper to be:encode as BERT and decode as GPT
5. where is the K, Q, V weights training happens? 
6. Can I interprete the meaning blobs inside K, Q, V? like CNN