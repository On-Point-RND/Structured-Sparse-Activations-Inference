python3 run_exps.py --model.path='NousResearch/Llama-2-7b-chat-hf' --pruning.sparsity_type='semi-structured_act_magnitude' --pruning.transformation_type='la_rosa'
python3 run_exps.py --model.path='Qwen/Qwen2.5-7B-Instruct' --pruning.sparsity_type='semi-structured_act_magnitude' --pruning.transformation_type='la_rosa' --pruning.target_modules='[o_proj,gate_proj,up_proj,down_proj]'
python3 run_exps.py --model.path='google/gemma-3-4b-it' --pruning.sparsity_type='semi-structured_act_magnitude' --pruning.transformation_type='la_rosa'

python3 run_exps.py --model.path='NousResearch/Llama-2-7b-chat-hf' --pruning.sparsity_type='semi-structured_act_magnitude' --pruning.transformation_type='la_rosa' --pruning.prune_n=4 --pruning.prune_m=8
python3 run_exps.py --model.path='Qwen/Qwen2.5-7B-Instruct' --pruning.sparsity_type='semi-structured_act_magnitude' --pruning.transformation_type='la_rosa' --pruning.prune_n=4 --pruning.prune_m=8 --pruning.target_modules='[o_proj,gate_proj,up_proj,down_proj]'
python3 run_exps.py --model.path='google/gemma-3-4b-it' --pruning.sparsity_type='semi-structured_act_magnitude' --pruning.transformation_type='la_rosa' --pruning.prune_n=4 --pruning.prune_m=8