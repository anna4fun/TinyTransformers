# Notes

## Overall mindset
1. Diffuse (no too focused): we don't want our model/results to be dependent on some part of the network too much, so we use Dropout; we do not want one token to be rely too much on one other token, so we implement normalization in attention.

## My timeline:
1. 2025/10/13 - 2025/10/17: Video watching
2. 2025/10/18: Load the data, implemented a toy example of weight matrix aggregation [code](code/weight_matrix_toy_example.py). 
And suddenly realized I need to add the `C (Channel)` dimension to the X.
I tried One-Hot-Encoding first because it's quickly, and then realized it doesn't make sense at all, and I really should use an embedding, here's [the notes about why](notes/where_shall_do_embedding.md)
3. 2025/10/19: Semi-copied the bigram model from the original repos' `bigram.py` file, get interested in what exactly does `nn.Embedding` means and have a bunch of Q&A with ChatGPT, here's the [notes about what does embedding do for training a Bigram model](notes/Is_the_embedding_table_the_training_target_for_BigramModel.md)
4. 2025/10/20 - 21: Structured the Bigram language model to be software dev style - there're source code, test code, train code and model checkpoints
5. 2025/10/22: initialize the gpt class and self attention head class
6. 2025/10/23: continue on self attention head
7. 2025/10/24: finish self attention, moving on to multi-head attention and forward pass, add drop-out block
8. 2025/10/27: test drop-out block, deep dive Q, K, V
9. 2025/10/28: add FeedForward and TransformerBlock classes, think about the dimension of input x, then deep dive into why does layer norm do column-wise normalization and why embedding = features 
10. 2025/10/29: train the GPT for the first time, debug the causal/lower triangle mask in the self attention

## Data Loading
The `get_batch` function gets a `batch_size` slices of x and y, 2 observations
1. The prediction task is auto-regressive, so `yb` is almost an exact copy of `xb` with just the 1st and last elements are different
2. Notice that `xb[0]` is a `1 x block_size` array, the `1` here means `n_embed = 1`, the reason being that we are tokenized at the character level, aka every character/token needs 1 integer to represent it. In more real setting, `n_embed > 1` and at those situations, `xb[0]` would be of size `n_embed x block_size`, eg. our current `x[0][0] = 51` would become `x[0][0] = [1,2,3,4]`
```
xb[0]:
tensor([51, 53, 57, 58,  1, 46, 39, 60, 43,  1, 53, 44,  1, 46, 47, 57,  1, 39,
        45, 43,  8,  0,  0, 28, 27, 24, 21, 36, 17, 26, 17, 31, 10,  0, 14, 63,
         1, 51, 63,  1, 61, 46, 47, 58, 43,  1, 40, 43, 39, 56, 42,  6,  0, 37,
        53, 59,  1, 53, 44, 44, 43, 56,  1, 46, 47, 51,  6,  1, 47, 44,  1, 58,
        46, 47, 57,  1, 40, 43,  1, 57, 53,  6,  1, 39,  1, 61, 56, 53, 52, 45,
         0, 31, 53, 51, 43, 58, 46, 47, 52, 45,  1, 59, 52, 44, 47, 50, 47, 39,
        50, 10,  1, 56, 43, 39, 57, 53, 52,  1, 51, 63,  1, 57, 53, 52,  0, 31,
        46, 53, 59, 50, 42,  1, 41, 46, 53, 53, 57, 43,  1, 46, 47, 51, 57, 43,
        50, 44,  1, 39,  1, 61, 47, 44, 43,  6,  1, 40, 59, 58,  1, 39, 57,  1,
        45, 53, 53, 42,  1, 56, 43, 39, 57, 53, 52,  0, 32, 46, 43,  1, 44, 39,
        58, 46, 43, 56,  6,  1, 39, 50, 50,  1, 61, 46, 53, 57, 43,  1, 48, 53,
        63,  1, 47, 57,  1, 52, 53, 58, 46, 47, 52, 45,  1, 43, 50, 57, 43,  0,
        14, 59, 58,  1, 44, 39, 47, 56,  1, 54, 53, 57, 58, 43, 56, 47, 58, 63,
         6,  1, 57, 46, 53, 59, 50, 42,  1, 46, 53, 50, 42,  1, 57, 53, 51, 43,
         1, 41, 53, 59], device='mps:0')
--------------------------
print(decoder(xb[0])):
most have of his age.
POLIXENES:
By my white beard,
You offer him, if this be so, a wrong
Something unfilial: reason my son
Should choose himself a wife, but as good reason
The father, all whose joy is nothing else
But fair posterity, should hold some cou   
--------------------------
yb[0]:
tensor([53, 57, 58,  1, 46, 39, 60, 43,  1, 53, 44,  1, 46, 47, 57,  1, 39, 45,
        43,  8,  0,  0, 28, 27, 24, 21, 36, 17, 26, 17, 31, 10,  0, 14, 63,  1,
        51, 63,  1, 61, 46, 47, 58, 43,  1, 40, 43, 39, 56, 42,  6,  0, 37, 53,
        59,  1, 53, 44, 44, 43, 56,  1, 46, 47, 51,  6,  1, 47, 44,  1, 58, 46,
        47, 57,  1, 40, 43,  1, 57, 53,  6,  1, 39,  1, 61, 56, 53, 52, 45,  0,
        31, 53, 51, 43, 58, 46, 47, 52, 45,  1, 59, 52, 44, 47, 50, 47, 39, 50,
        10,  1, 56, 43, 39, 57, 53, 52,  1, 51, 63,  1, 57, 53, 52,  0, 31, 46,
        53, 59, 50, 42,  1, 41, 46, 53, 53, 57, 43,  1, 46, 47, 51, 57, 43, 50,
        44,  1, 39,  1, 61, 47, 44, 43,  6,  1, 40, 59, 58,  1, 39, 57,  1, 45,
        53, 53, 42,  1, 56, 43, 39, 57, 53, 52,  0, 32, 46, 43,  1, 44, 39, 58,
        46, 43, 56,  6,  1, 39, 50, 50,  1, 61, 46, 53, 57, 43,  1, 48, 53, 63,
         1, 47, 57,  1, 52, 53, 58, 46, 47, 52, 45,  1, 43, 50, 57, 43,  0, 14,
        59, 58,  1, 44, 39, 47, 56,  1, 54, 53, 57, 58, 43, 56, 47, 58, 63,  6,
         1, 57, 46, 53, 59, 50, 42,  1, 46, 53, 50, 42,  1, 57, 53, 51, 43,  1,
        41, 53, 59, 52], device='mps:0')
--------------------------
print(decoder(yb[0])):
ost have of his age.
POLIXENES:
By my white beard,
You offer him, if this be so, a wrong
Something unfilial: reason my son
Should choose himself a wife, but as good reason
The father, all whose joy is nothing else
But fair posterity, should hold some coun
```

## 
