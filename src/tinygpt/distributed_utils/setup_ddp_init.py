import os
import torch
from torch.distributed import init_process_group

def init_ddp(ddp):
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"

        # Initialize DDP communication (NCCL for CUDA)
        init_process_group(backend='nccl')

        # Get DDP environment variables (set by torchrun/launcher)
        ddp_rank = int(os.environ['RANK'])   # Global rank (unique across all nodes)
        ddp_local_rank = int(os.environ['LOCAL_RANK'])  # Local GPU index (0,1,2,3 on current machine)
        ddp_world_size = int(os.environ['WORLD_SIZE'])  # Total number of processes/GPUs

        # Assign the process to its dedicated GPU
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)

        # Only the master_process will do logging, checkpointing etc.
        master_process = ddp_rank == 0
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    return device, ddp_rank, ddp_local_rank, ddp_world_size, master_process