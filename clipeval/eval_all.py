import importlib
import os
import json

from pathlib import Path
from dataclasses import dataclass


baselines = [
    ("metaclip2_s16", 'ViT-S-16-worldwide@WorldWideCLIP', 'metaclip2_worldwide', "facebook/xlm-v-base"),
]


eval_modules = [
    ("slip", "clipeval.slip.eval_slip"),
    ("xm3600", "clipeval.xm3600.eval_xm3600"),
    ("cvqa", "clipeval.cvqa.eval_cvqa"),
]


def load_model(model_name, pretrained, tokenizer_name):
    if '@' not in model_name:
        # config open_clip; no installation required.
        if 'external/open_clip/src' not in sys.path:
            sys.path.append('external/open_clip/src')
        from external.open_clip.src import open_clip
        model, _, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(tokenizer_name)
    else:  # metaclip models
        from src.mini_clip.factory import create_model_and_transforms, get_tokenizer
        model, _, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = get_tokenizer(tokenizer_name)
    model.cuda()
    model.eval()
    return model, preprocess_val, tokenizer


def eval_all(bench_id):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import json
    import sys
    if "." not in sys.path:
        sys.path.append(".")

    config_name, model_name, pretrained, tokenizer_name = baselines[bench_id]

    # create or reuse metaclip training log dir.
    Path(f"logs/{config_name}").mkdir(parents=True, exist_ok=True)
    print(config_name)

    results = {}

    model = None

    for benchname, module_path in eval_modules:
        print(f"Eval {benchname}")
        result_json = f"logs/{config_name}/{benchname}.json"
        module = importlib.import_module(module_path)

        if not Path(result_json).exists():
            if model is None:
                model, preprocess_val, tokenizer = load_model(model_name, pretrained, tokenizer_name)    
            getattr(module, "main")(model, preprocess_val, tokenizer, result_json)

        # dump results to `results`.
        getattr(module, "parse_results")(results, result_json)

    print(json.dumps(results, indent=4))


if __name__ == '__main__':

    import sys
    if True:
        if len(sys.argv) == 2:  # python clipeval/eval_all.py 0  # 0 means first task
            eval_all(int(sys.argv[1]) )
        for bench_id in range(len(eval_modules)):
            eval_all(bench_id)
    else:
        offsets = list(range(len(baselines)))
        import os
        import submitit
        import sys

        partition = "your_partition"
        slurm_additional_parameters={"account": partition, "qos": "your_qos"}

        params = dict(
            name=f"clipeval",
            gpus_per_node=1,
            mem_gb=130,
            cpus_per_task=4,
            nodes=1,
            slurm_partition='learn',
            timeout_min=4320,
            slurm_additional_parameters=slurm_additional_parameters
        )

        executor = submitit.AutoExecutor(
            folder="submitit/%j"
        )
        executor.update_parameters(**params)

        jobs = []
        with executor.batch():
            for offset in offsets:
                job = executor.submit(
                    eval_all,
                )
                jobs.append(job)

        if len(jobs) > 0:
            print(partition, len(jobs), jobs[0].job_id, jobs[-1].job_id)
