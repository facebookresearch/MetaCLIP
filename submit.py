# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import uuid
from pathlib import Path

import sys
sys.path.append("./")

import src.training.main as main
import submitit


def parse_args():
    parser = argparse.ArgumentParser("Submitit")
    parser.add_argument("config_name", type=str, help="name of the config.")
    parser.add_argument("--ngpus", default=None, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=None, type=int, help="Number of nodes to request")
    parser.add_argument("--resume", default="", type=str, help="resume a checkpoint.")
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--partition", default="learnlab", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Request 32G V100 GPUs")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")
    args = parser.parse_args()
    return args


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/openclip")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.args.config.dist_url = get_init_file().as_uri()

    def __call__(self):
        import sys
        sys.path.append("./")
        import src.training.main as main
        self._setup_gpu_args()
        main.main(self.args.config)

    def checkpoint(self):
        import os
        import submitit

        self.args.config.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.config.output_dir, "checkpoints", "epoch_latest.pt")
        if os.path.exists(checkpoint_file):
            self.args.config.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        if self.args.ngpus >= 1:
            self.args.config.local_rank = job_env.local_rank
            self.args.config.rank = job_env.global_rank
            self.args.config.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main(args):
    if args.job_dir == "":
        args.job_dir = get_shared_folder()

    assert args.job_dir != ""
    if os.path.exists(args.job_dir) and len(args.resume) == 0 and not hasattr(args.config, "eval"):
        raise ValueError(f"{args.job_dir} existed, rm -rf {args.job_dir} ?")

    args.job_dir = Path(args.job_dir) / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node, # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name=args.config.name)

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id, "@", str(args.job_dir).replace("%j", job.job_id))


def submit():
    args = parse_args()
    from configs import search_config
    from copy import deepcopy

    config = search_config(args.config_name)
    _args = deepcopy(args)
    if len(args.resume):
        checkpoint_file = os.path.join(config.output_dir, "checkpoints", args.resume)
        args.resume = checkpoint_file
        config.resume = checkpoint_file

    setattr(_args, "config", config)
    if args.ngpus is not None:
        _args.ngpus = args.ngpus
    elif hasattr(config, "ngpus"):
        _args.ngpus = config.ngpus
    else:
        raise ValueError("must specify ngpus in arg or config.")
    if args.nodes is not None:
        _args.nodes = args.nodes
    elif hasattr(config, "nodes"):
        _args.nodes = config.nodes
    else:
        raise ValueError("must specify ngpus in arg or config.")
    _args.job_dir = config.output_dir
    main(_args)


if __name__ == "__main__":
    submit()
