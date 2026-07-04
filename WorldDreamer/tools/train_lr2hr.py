import os
import sys
import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch

from mmdet3d.datasets import build_dataset
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

# fmt: off
# bypass annoying warning
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# fmt: on

sys.path.append(".")  # noqa
from dataset import *
from projects.dreamer.utils.common import load_module


def set_logger(global_rank, logdir):
    if global_rank == 0:  # already set for main process
        return
    logging.info(f"reset logger for {global_rank}")
    root = logging.getLogger()
    root.handlers.clear()  # we reset logger for other processes
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    # to logger
    file_path = os.path.join(logdir, f"train.{global_rank}.log")
    handler = logging.FileHandler(file_path)
    handler.setFormatter(formatter)
    root.addHandler(handler)


@hydra.main(version_base=None, config_path="../configs", config_name="config_nus_448")
def main(cfg: DictConfig):
    if cfg.debug:
        import debugpy
        debugpy.listen(cfg.get('debug_port', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print('Attached, continue...')

    # setup logger
    # only log debug info to log file
    logging.getLogger().setLevel(logging.DEBUG)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler) or cfg.try_run:
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
    # handle log from some packages
    logging.getLogger("shapely.geos").setLevel(logging.WARN)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("accelerate.tracking").setLevel(logging.INFO)
    logging.getLogger("numba").setLevel(logging.WARN)
    logging.getLogger("PIL").setLevel(logging.WARN)
    logging.getLogger("matplotlib").setLevel(logging.WARN)
    setattr(cfg, "log_root", HydraConfig.get().runtime.output_dir)

    # multi process context
    # since our model has randomness to train the uncond embedding, we need this.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.accelerator.gradient_accumulation_steps,
        mixed_precision=cfg.accelerator.mixed_precision,
        log_with=cfg.accelerator.report_to,
        project_dir=cfg.log_root,
        kwargs_handlers=[ddp_kwargs],
    )
    set_logger(accelerator.process_index, cfg.log_root)
    set_seed(cfg.seed)
    
    # LR = Low Resolution, HR = High Resolution
    if cfg.stage == "LR":
        logging.info("Starting Low-Resolution (LR) pre-training stage.")
        train_dataset = build_dataset(OmegaConf.to_container(cfg.dataset.data.train, resolve=True))
        val_dataset = build_dataset(OmegaConf.to_container(cfg.dataset.data.val, resolve=True))
    elif cfg.stage == "HR":
        logging.info("Starting High-Resolution (HR) fine-tuning stage.")
        train_dataset = build_dataset(OmegaConf.to_container(cfg.dataset.data.train, resolve=True))
        val_dataset = build_dataset(OmegaConf.to_container(cfg.dataset.data.val, resolve=True))
        
        # Resume from LR checkpoint if specified
        if cfg.resume_from_checkpoint:
            logging.info(f"Resuming from LR checkpoint: {cfg.resume_from_checkpoint}")
        else:
            logging.warning("HR stage started without specifying an LR checkpoint.")
            
    else:
        raise ValueError(f"Invalid stage specified: {cfg.stage}. Must be 'LR' or 'HR'.")

    # runner
    if cfg.resume_from_checkpoint and cfg.resume_from_checkpoint.endswith("/"):
        cfg.resume_from_checkpoint = cfg.resume_from_checkpoint[:-1]
    runner_cls = load_module(cfg.model.runner_module)
    runner = runner_cls(cfg, accelerator, train_dataset, val_dataset)
    runner.set_optimizer_scheduler() 
    runner.prepare_device()

    # tracker
    if accelerator.is_main_process:
        accelerator.init_trackers(f"tb-{cfg.task_id}-{cfg.stage}", config=None)
    
    logging.info(f"Starting run for stage: {cfg.stage}")
    runner.run()
    logging.info(f"Stage {cfg.stage} finished successfully.")
    
    if accelerator.is_main_process and cfg.stage == "LR":
        output_dir = HydraConfig.get().runtime.output_dir
        
        info_file_path = f"/tmp/run_info_{cfg.runner.run_id}.txt"

        with open(info_file_path, "w") as f:
            f.write(output_dir)
            
        last_checkpoint_path = runner.get_last_checkpoint_path()
        
        with open(os.path.join(output_dir, "last_checkpoint.txt"), "w") as f:
            f.write(last_checkpoint_path)
        
        logging.info(f"Last checkpoint path saved to {output_dir}/last_checkpoint.txt")

if __name__ == "__main__":
    main()
