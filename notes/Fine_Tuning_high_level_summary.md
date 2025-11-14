# A High Level Summary of Finetuning

Pre-trained language models like GPT2 and GPT3 are trained to perform next-word predictions, 
eg. given a prefix of sentence "Hello I am an LLM", the model will start generate the next word followed by this prefix sentence and the one after next, so on so force. 

But this next-word prediction action doesn't seem to be very useful in a real-world settings because this is not how we as human beings give instructions, 
instead, human instructions are usually articulated as:

1. Summarize an article 
2. Fix grammars and rewrite a document
3. Translate a document from English to Portuguese
4. Answer a question

Guess what: these tasks are still next-word predictions! 
The only difference is that the training set is composed with structures rather than free-form text fetched directly from Wikipedia or Reddit. 

Take the summarization task as an example, the training set contains pairs of `inputs`(which is an article) and `target` (which is human-written summary), 
and we refine the model's parameters so that when the prefix is "Summarize + input article" the model generates the next-tokens that are very closed to the human-written summary.

With this structure of input and target, it became a traditional supervised learning task, 
that's why the industry named this training step as Supervised Fine-tuning.

## Supervised Fine-Tuning (SFT)
### The high level 
#### 1. Building the training set
The training set of SFT have certain structures.

For summarization task, the `input` article and the `target` human-written summary are wrapped in 
an **instructional template**:
```
You are a helpful assistant that summarizes articles.

Article:
{ARTICLE_TEXT}

Summary:
{SUMMARY_TEXT}

```
The "You are a helpful assistant that summarizes articles." is what we called a prompt, which is later used in prompt engineering.

#### 2. Train the instructional sequence with loss
The tokenizer will turn the instructional template as the following:
```
[You are a helpful ... Article:\n]  a1 a2 a3 ... aN  [\nSummary:\n]  s1 s2 s3 ... sM
```
So we know clearly the tokens before `[\nSummary:\n]` are the inputs and the tokens `s1 s2 s3 ... sM` are the target we are going to calculate loss with.

```
input_ids = tokenizer(full_text, ...).input_ids

# Suppose we know where the summary starts:
summary_start = find_summary_start_index(input_ids)

labels = [-100] * len(input_ids) # todo: why label = ignore_index (e.g. -100) makes no loss is computed?
labels[summary_start:] = input_ids[summary_start:]  # predict summary tokens only

outputs = model(input_ids=torch.tensor([input_ids]), # feed the whole sentence
                labels=torch.tensor([labels]))       # calc loss on summary tokens only
loss = outputs.loss
loss.backward()
optimizer.step()
```


#### 3. Now we can use it - Prompts only vs. Few-Shot


#### 4. The mysteries
1. Does the model knows keywords such as 'summarize', 'refine grammar', etc? If so, how to achieve this?
2. There is a sorting in the context understanding and text generation, especially in question answering task (which is essentially searching and ranking) 


### LoRA, the Swiss knife
LoRA, low rank adaptation of LLMs, makes training light and fast by keeping most parts of the pre-trained model frozen and injects low-rank adaptors into the attention and MLP layers.

#### LoRA in plain English
Think of LoRa training a small correction that we add to the original attention and MLP parameter matrix, and it's very cheap to add these corrections.

Recall that we want to the model to learn from the new training set, here "learn" means updating the model's parameters, 
which is essentially updating the weights like:  

$$ W_{fine\_tuned} = W_{original} + \delta W$$

with $\delta W$ as small incremental changes to the original weights, $W_{fine\_tuned},  W_{original} , \delta W$ all of shape $[d_{in},  d_{out}]$ . 

Now came the sharp blades part: If we train $\delta W$ as a whole, that means we are calculating $d_{in} \times d_{out}$ which is a lot of parameters. 
However, LoRA says we can break down $\delta W$ into lower dimensional subspaces like this
$$ \delta W = B \times A $$
where B is of shape $d_{in} \times r$ and B of shape $r \times d_{out}$, $r$ is the LoRA rank parameter and it's usually taking very small values (eg. 4, 8 ,16).
Now the number of parameters waiting to be trained goes down from $d_{in} \times d_{out}$ to $r \times (d_{in} + d_{out})$.

#### How big is the reduction:

Take GPT2's attention layer as an example, `c_attn_0` is of shape $[768, 2304]$, if we take LoRA rank $r = 16$,
then the parameters waiting to be trained goes down from $768\times2304=1,769,472$ into $16 \times (768+2304) = 49,152$, that's 36x reduction!

#### Wrapping the LoRA correction in the Forward function
Freeze (aka no gradients)




"Low-Rank" refers to a smaller dimension of matrix compared with the original attention/MLP parameters. 


We can multitask the LoRa training by simply mixing up the tasks templates, eg. translation, Q&A, rewriting, and plug-in a few thousands of golden set examples for each task categories.


### The Do's and Don't of Fine-tuning



## The risks and benefits

### How we get the training set
#### Where do we get a golden set of human-written summaries as Ground Truth

### How can we make the most out of fine-tuned models
#### Prompt engineering


### TODO
1. a table of typical training sets for common tasks including summarization, fix grammar, translation, question answering

2. prompt understanding (is this encoding?) and is prompt engineering basically taking examples from the training set
3. does RAG basically keep a local/temporary storage of the input article, the model learn the context from this article which is generate the KV cache or the vector store, and then fetch the prompt to know what to generate?
4. Why calling gpt2 as "causal ML"? the lower triangle mask is called the "causal mask". what does this has to do with "causal"? the left happens so it lead to the next token/action on the right?