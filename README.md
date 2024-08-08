# MiM-SRCO: Momentum-integrated Multi-task Stock Recommendation with Converge-based Optimization
## An Official Implementation for KDD-25 submission

### Environment
Create a Python 3.8 environment using [href](requirements.txt)


### Reproduce the result
change [backbone] to [GRU, LSTM, GATs, RSR, HIST]
```
# For CSI300 dataset
python exp/mtl_training.py --method our_method --device cuda:0 --ndcg approx --adaptive_k True --loss_type mixed --outdir [target_location] --mtm_source_path --model_name [backbone] --class_weight_method square
# For CSI100 dataset
python exp/mtl_training.py --method our_method --device cuda:0 --ndcg approx --adaptive_k True --loss_type mixed --outdir [target_location] --mtm_source_path ./data/csi100_mtm.pkl --model_name [backbone] --class_weight_method square
```

### MTL Baselines
change [baseline] to [nashmtl, cagrad, dbmtl, uniw]
```
# For CSI300 dataset
python exp/mtl_training.py --method [baseline] --device cuda:0 --ndcg approx --adaptive_k True --loss_type mixed --outdir [target_location] --mtm_source_path --model_name [backbone] --class_weight_method square
# For CSI100 dataset
python exp/mtl_training.py --method [baseline] --device cuda:0 --ndcg approx --adaptive_k True --loss_type mixed --outdir [target_location] --mtm_source_path ./data/csi100_mtm.pkl --model_name [backbone] --class_weight_method square
```

### Single task learning
```
# For CSI300 dataset
python exp/regression_training.py --model_name [backbone] --outdir [target_location] --repeat 3 --device cuda:0
python exp/classification_training.py --model_name [backbone] --outdir [target_location] --repeat 3 --device cuda:0
# For CSI100 dataset
python exp/regression_training.py --model_name ALSTM --outdir [target_location] --repeat 3 --device cuda:1 --mtm_source_path ./data/csi100_mtm.pkl
python exp/classification_training.py --model_name ALSTM --outdir [target_location] --repeat 3 --device cuda:1 --mtm_source_path ./data/csi100_mtm.pkl
```

### To transfer to rise-or-fall task, add ```--mtm_column mtm0101```
### To use cross-entropy or pair-wise loss function, add ```--loss_type cross-entropy``` or  ```--loss_type pair-wise```


Thanks to the following open-source repositories:
1. https://github.com/Wentao-Xu/HIST
2. https://github.com/AvivNavon/nash-mtl
3. https://github.com/thuml/Time-Series-Library
4. https://github.com/median-research-group/LibMTL