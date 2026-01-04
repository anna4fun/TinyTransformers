import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def train_loop(cfg: GPT2DataConfig):
        # Initialize DDP
        if cfg.ddp:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            cfg.rank = local_rank
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")

        # Setup logger (critical: call after DDP rank is set)
        logger = setup_logger(cfg)

        # Log training config (only rank 0)
        if cfg.rank == 0:
            logger.info("=" * 50)
            logger.info(f"Starting Training with Config: {cfg}")
            logger.info(f"Checkpoint Dir: {cfg.checkpoint_dir}")
            logger.info(f"Resume Checkpoint: {cfg.resume_checkpoint or 'None'}")
            logger.info("=" * 50)

        # Build DataLoaders
        dataloaders = make_ddp_dataloader(cfg)
        train_dl = dataloaders["train_dl"]
        train_dataset = dataloaders["train_dataset"]

        # Initialize model/optimizer/scheduler
        logger.info(f"Initializing model on GPU {cfg.rank}")
        model = torch.nn.Sequential(
            torch.nn.Embedding(50257, 768),
            torch.nn.Linear(768, 50257)
        ).to(cfg.rank)

        if cfg.ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.rank])
            logger.info(f"Wrapped model in DDP on GPU {cfg.rank}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        # Load checkpoint
        resume_shard_idx, resume_seq_idx, start_epoch, start_loss = load_training_checkpoint(
            cfg, model, optimizer, scheduler
        )
        train_dataset.resume_shard_idx = resume_shard_idx
        train_dataset.resume_seq_idx = resume_seq_idx
        logger.info(f"Resumed from shard {resume_shard_idx}, epoch {start_epoch} (GPU {cfg.rank})")

        # Training loop
        current_shard_idx = resume_shard_idx
        for epoch in range(start_epoch, 10):
            if cfg.ddp:
                train_dl.sampler.set_epoch(epoch)
                logger.info(f"Starting Epoch {epoch} on GPU {cfg.rank} (shuffled sampler)")

            pbar = tqdm(train_dl, desc=f"Epoch {epoch} (GPU {cfg.rank})", disable=cfg.rank != 0)
            for batch_idx, (x, y) in enumerate(pbar):
                x = x.to(cfg.rank, non_blocking=True)
                y = y.to(cfg.rank, non_blocking=True)

                # Forward/backward pass
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Update progress bar
                pbar.set_postfix({"loss": loss.item()})

                # Log batch loss (optional: log every 100 batches to reduce noise)
                if batch_idx % 100 == 0:
                    logger.debug(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} (GPU {cfg.rank})")

                # Get current shard index
                current_seq = train_dataset._get_globally_unique_indices()[batch_idx]
                current_shard_idx = current_seq[0]

                # Save checkpoint every N shards (rank 0 only)
                if (current_shard_idx % cfg.checkpoint_every_n_shards == 0) and (cfg.rank == 0):
                    save_training_checkpoint(
                        cfg, model, optimizer, scheduler,
                        shard_idx=current_shard_idx,
                        seq_idx=current_seq[1],
                        epoch=epoch,
                        loss=loss.item()
                    )
                    logger.info(f"Saved checkpoint for shard {current_shard_idx} (Epoch {epoch})")

        # Cleanup
        logger.info(f"Training completed on GPU {cfg.rank}")
        if cfg.ddp:
            dist.destroy_process_group()

    # -------------------------- Update Checkpoint Functions with Logger --------------------------
    def save_training_checkpoint(
            cfg: GPT2DataConfig,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
            shard_idx: int,
            seq_idx: int,
            epoch: int,
            loss: float
    ):
        logger = logging.getLogger("fineweb_training")  # Get global logger
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            cfg.checkpoint_dir,
            f"ckpt_shard_{shard_idx}_epoch_{epoch}.pt"
        )

        try:
            model_state = model.module.state_dict() if cfg.ddp else model.state_dict()
            checkpoint = {
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "resume_shard_idx": shard_idx,
                "resume_seq_idx": seq_idx,
                "epoch": epoch,
                "loss": loss,
                "cfg": cfg
            }
            torch.save(checkpoint, checkpoint_path)

            # Save resume state
            resume_state = {"resume_shard_idx": shard_idx, "resume_seq_idx": seq_idx}
            with open(os.path.join(cfg.checkpoint_dir, "resume_state.json"), "w") as f:
                json.dump(resume_state, f)

            logger.info(f"Checkpoint saved successfully: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}", exc_info=True)  # Log full error trace

    def load_training_checkpoint(
            cfg: GPT2DataConfig,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    ) -> Tuple[int, int, int, float]:
        logger = logging.getLogger("fineweb_training")
        if not cfg.resume_checkpoint or not os.path.exists(cfg.resume_checkpoint):
            logger.warning("No valid resume checkpoint found — starting fresh")
            return 0, 0, 0, 0.0

        try:
            checkpoint = torch.load(cfg.resume_checkpoint, map_location=f"cuda:{cfg.rank}")
            # Load model/optimizer/scheduler
            if cfg.ddp:
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler and checkpoint["scheduler_state_dict"]:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            resume_shard_idx = checkpoint["resume_shard_idx"]
            resume_seq_idx = checkpoint["resume_seq_idx"]
            epoch = checkpoint["epoch"]
            loss = checkpoint["loss"]

            logger.info(f"Successfully loaded checkpoint: {cfg.resume_checkpoint}")
            logger.info(f"Resumed state: shard {resume_shard_idx}, epoch {epoch}, loss {loss:.4f}")
            return resume_shard_idx, resume_seq_idx, epoch, loss
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}", exc_info=True)
            raise  # Re-raise error to stop training (critical failure)


if __name__ == '__main__':
    unittest.main()
