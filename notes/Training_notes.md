# Training notes

## High level
1. The GPT2 Language Model
   2. Byte-pair encoding tokenization
   3. Build contextual embedding with the attention mechanism
   3. The transformer block data flow: idx -> token+positional embedding -> 12 layers of Block (Attention + MLP + LayerNorm)  -> Final layer norm -> projection back to vocabulary space -> softmax for next token prediction
4. The Training
   5. Prepare data loader
   6. Initialize the GPT2 model parameters with down-scaled standard deviation. (#stability)
      7. (GPT2 only) (#efficiency) token embedding share the same weights with the final projection head's weights, this aligns the distribution of inputs and outputs
   7. (Forward) Feed sequences of data into GPT2 language model to get the logits
   7. (Backward) Use cross-entropy loss (negative averaged log-likelihood) for gradient descent 

**2025/12/03**

With the Experiment Config (small layers and n_embd) Overfit a single batch of data would land the training loss and valid loss both to 9. 
Iterate over 50 batches doesn't change the loss too much. The loss indicates underfitting.
What make a difference was to switch to the GPT2DataConfig (with bigger n_embd,  n_layers), the valid loss goes to 6.8, much better. Total training takes 24 mins which is very slow.


**2025/12/05**
Add the GPT2 (1) weight sharing scheme between token embedding weights and the lm_head weights and (2) initialization tricks.
(1) improved training efficiency by reducing the number of parameters to be trained by 30% (size of wte is 768*50257= 38M, which is 38M/124M = 30% of the 124M parameters). However the total training time increased and the valid loss also increased a little bit.
(2) initialization tricks does decrease the loss a little bit