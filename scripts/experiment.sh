mkdir ./output/single_training
# retrain regression
python exp/regression_training.py --model_name PatchTST --outdir ./output/single_training/PatchTST_regression --repeat 3 --device cuda:0
python exp/regression_training.py --model_name GRU --outdir ./output/single_training/GRU_regression --repeat 3 --device cuda:0
python exp/regression_training.py --model_name RSR --outdir ./output/single_training/RSR_regression --repeat 3 --device cuda:0
python exp/regression_training.py --model_name HIST --outdir ./output/single_training/HIST_regression --repeat 3 --device cuda:0

# new backbone single training
python exp/regression_training.py --model_name LSTM --outdir ./output/single_training/LSTM_regression --repeat 3 --device cuda:0
python exp/regression_training.py --model_name GATs --outdir ./output/single_training/GATs_regression --repeat 3 --device cuda:0
python exp/classification_training.py --model_name LSTM --outdir ./output/single_training/LSTM_classification --repeat 3 --device cuda:0
python exp/classification_training.py --model_name GATs --outdir ./output/single_training/GATs_classification --repeat 3 --device cuda:0


#mkdir .output/our_method
#python exp/mtl_training.py --method our_method --device cuda:0 --ndcg approx --adaptive_k True --loss_type mixed --outdir ./output/our_method/GATs_ourmethod --model_name GATs --class_weight_method square
