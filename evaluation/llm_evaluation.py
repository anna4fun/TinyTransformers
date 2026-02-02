import os
import sys
from threading import local
from tinygpt.configs.config import GPT2DataConfig
from tinygpt.logger.setup_logger import setup_logger
from tinygpt.models.gpt2 import GPT2
import torch
from tinygpt.evaluation_utils.eval_loss import loss_evaluate
from tinygpt.data_loaders.gpt2_data_loader import make_ddp_dataloader

def llm_evaluation(model, checkpoint_name=None):
    # detect device
    device = 'mps'
    if torch.cuda.is_available():
        device = 'cuda'
    # load LLM config and initialize model
    config = GPT2DataConfig(vocab_size=50304, batch_size=16,  device=device, num_workers=8, resume_checkpoint=True, seed=1039)
    model = model(config)

    # setup logger
    logger = setup_logger(cfg=config, train_name="GPT2 Evaluation", local_rank=0)
    logger.info(f"Starting device {device}")

    # load model checkpoints
    if checkpoint_name:
        logger.info(f"Evaluating model from {checkpoint_name}")
        checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_name)
        if config.resume_checkpoint and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Load checkpoint success")
            except Exception as exc:
                logger.info(f"Fail to load checkpoint")
                logger.exception(exc)
    # Move model to devices
    model.to(device)
      
    # Load FineWeb-Edu Test set
    dl_generator = torch.Generator()
    dl_generator.manual_seed(config.seed)
    dl = make_ddp_dataloader(config,g=dl_generator)
    test_dl = dl["test_dl"]                

    # Loss and Perplexity
    eval_batch_size = 7000
    logger.info(f"test_dl batche size: {len(test_dl)}, evaluating {eval_batch_size} batches with new seed {config.seed}")
    with torch.autocast(device_type = device,dtype=torch.bfloat16):
        test_loss, test_perplexity = loss_evaluate(
            model, valid_loader=test_dl, device=device, 
            perplexity_required=True, max_val_batches=eval_batch_size)
    logger.info(f"Test loss: {test_loss}, test perplexity: {test_perplexity}")
    
    # HellaSwag: common sense

    # Generation

if __name__ == '__main__':
    # My own GPT2 model
    llm_evaluation(model=GPT2, checkpoint_name = f"ckpt_gpt2_epoch_1_19072.pt")
    # HuggingFace GPT2 model
    sys.exit()
