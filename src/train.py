# # create a PyTorch optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#
# for iter in range(max_iters):
#     # every once in a while evaluate the loss on train and val sets
#     if iter % eval_interval == 0:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
#     # sample a batch of data
#     xb, yb = get_batch('train')
#     # evaluate the loss
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
#
# # generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decoder(m.generate(context, max_new_tokens=500)[0].tolist()))