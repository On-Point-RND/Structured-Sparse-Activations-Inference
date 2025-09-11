python3 run_exps.py --pruning.sparsity_type='unstructured_act_magnitude'  --pruning.transformation_type=none       --pruning.sparsity_ratio=0.5
python3 run_exps.py --pruning.sparsity_type='unstructured_act_magnitude'  --pruning.transformation_type='variance' --pruning.sparsity_ratio=0.5
python3 run_exps.py --pruning.sparsity_type='unstructured_act_magnitude'  --pruning.transformation_type='shift'    --pruning.sparsity_ratio=0.5
python3 run_exps.py --pruning.sparsity_type='unstructured_clact_pruner'  --pruning.transformation_type=none    --pruning.sparsity_ratio=0.5
python3 run_exps.py --pruning.sparsity_type='unstructured_amber_pruner'  --pruning.transformation_type=none    --pruning.sparsity_ratio=0.5
