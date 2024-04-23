# Copyright (c) Meta Platforms, Inc. and affiliates

import sys
sys.path.append("src")
sys.path.append("./")
import os

import numpy as np
import pandas as pd
import json
import torch
from clipeval.eval_zeroshot import mean_per_class, accuracy, roc_auc

from scipy.special import softmax

def evaluate_dataset(d,acc_or_outputs):
    if d in ['FGVCAircraft', 'OxfordPets', 'Caltech101', 'Flowers102']:
        metric = mean_per_class(*acc_or_outputs)
    elif d == 'Kinetics700':
        top1, top5 = accuracy(*acc_or_outputs, topk=(1, 5))
        metric = (top1 + top5) / 2
        metric = metric.item()
    elif d == 'HatefulMemes':
        metric = roc_auc(*acc_or_outputs)
    else:
        pred = acc_or_outputs[0].argmax(dim=1)
        correct = pred.eq(acc_or_outputs[1]).sum()
        metric = correct.item() / float(pred.size(0)) * 100.0
    return metric

def process(dataset, num_cluster, result_path, ensemble_weight):
    outputs = []
    units_acc = []
    result = {}
    for i in range(num_cluster):
        data = torch.load(os.path.join(result_path,'{}_pred-{}.pth'.format(dataset,i)),map_location='cpu')
        outputs.append(data)
        units_acc.append(evaluate_dataset(dataset,(data['logits'],data['targets']))) 
        
    result = {f'expert-{i}':acc for i,acc in enumerate(units_acc)}
    result['expert-max'] = max(units_acc)
    all_logits = torch.stack([item['logits'] for item in outputs])
    result['all_unit'] = evaluate_dataset(dataset,(all_logits.mean(dim=0), data['targets'])) 
    result['ensemble'] = evaluate_dataset(dataset,((all_logits * torch.from_numpy(ensemble_weight).view(-1,1,1)).sum(dim=0), data['targets'])) 

    return result

def main(opts):

    csv_path = os.path.join(opts.output_dir,'result.csv')
    tasks = json.load(open(
        '{}/clipeval/dataset_catalog.json'.format('..' if os.path.abspath('.').endswith('mode') else '.'),'r'
    ))
    fine_fn = os.path.join(opts.hrchy_assign,opts.dist_type,'F{}.pth'.format(opts.mode_fine))
    fine_cluster = torch.load(fine_fn)['center'].cpu()
    hrchy_fn = os.path.join(opts.hrchy_assign,opts.dist_type,'F{}-C{}.pth'.format(opts.mode_fine,opts.mode_size)) 
    hrchy_assign = torch.load(hrchy_fn)['assign'].numpy()

    results = {'dataset':[],'num_classes':[]}
    for dataset in tasks:

        # prepare raw distance between task embedding and cluster centers
        close_fn = os.path.join(opts.metadata_dir,f'{dataset}.json')
        if os.path.exists(close_fn):
            with open(close_fn,'r') as json_file:
                fine_closeness = json.load(json_file)
            [dist,assign] = fine_closeness
        else:
            task_embedding = torch.load(os.path.join(opts.metadata_dir,f'{dataset}.pth'))
            # top 1 filtering
            dist,assign = torch.cdist(task_embedding,fine_cluster).min(dim=-1)
            dist,assign = dist.tolist(),assign.tolist()
            with open(close_fn,'w') as json_file:
                json.dump([dist,assign],json_file)
        
        # grouping along coarse cluster
        n_class = len(assign)
        hrchy_avg = hrchy_assign[np.array(assign)]
        soft_avg = {key: np.array(dist)[hrchy_avg==key]  for key in np.unique(hrchy_assign).tolist()}

        # Ensembling
        weight_soft_unit = np.zeros(len(soft_avg))
        for key,item in soft_avg.items():
            if len(item) > 0:
                sharpen_add = np.exp(0.5-np.sqrt(n_class)) if n_class < 10 else 0.0
                sharpen_mul = opts.smooth_weight[0] * np.log10(max(10,n_class-opts.too_many_class))
                weight_soft_unit[key] = np.exp((sharpen_add-item) * sharpen_mul).sum()
        weight_soft_unit = softmax(weight_soft_unit/opts.smooth_weight[1])
        unit_result = process(dataset, opts.mode_size, opts.result_dir, weight_soft_unit)

        # summary
        results['dataset'].append(dataset)
        results['num_classes'].append(n_class)
        for key in unit_result:
            if key not in results:
                results[key] = []
            results[key].append(unit_result[key])
    
    # stat average and report
    for key in results:
        if key == 'dataset':
            results[key].append('average')
        else:
            results[key].append(np.mean(results[key]))
    result_df = pd.DataFrame.from_dict(results)
    result_df.to_csv(csv_path)
    print('Ensembling Done, please check',  csv_path)

if __name__ == "__main__":
    from configs_mode import search_config
    config = search_config(sys.argv[1])

    config.output_dir = os.path.dirname(config.output_dir)
    config.result_dir = os.path.join(config.output_dir,'eval_outputs')
    config.metadata_dir = sys.argv[2]
    config.smooth_weight = [5.0,8.0]
    config.too_many_class = 200

    main(config)

