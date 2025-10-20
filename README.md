# Build a Tiny Transformers (Decoder Only)
This repository contains my code and notes for implementing a tiny decoder only transformers, 
following Andrej Karpathy's YouTube video [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6576s). 
This repo was forked from the lecture's [repo](https://github.com/karpathy/ng-video-lecture)

My timeline:
1. 2025/10/13 - 2025/10/17: Video watching
2. 2025/10/18: Load the data, implemented a toy example of weight matrix aggregation [code](weight_matrix_toy_example.py). 
And suddenly realized I need to add the `C (Channel)` dimension to the X.
I tried One-Hot-Encoding first because it's quickly, and then realized it doesn't make sense at all, and I really should use an embedding, here's [the notes about why](where_shall_do_embedding.md)
3. 2025/10/19: Semi-copied the bigram model from the original repos' `bigram.py` file, get interested in what exactly does `nn.Embedding` means and have a bunch of Q&A with ChatGPT, here's the [notes about what does embedding do for training a Bigram model](Is_the_embedding_table_the_training_target_for_BigramModel.md)