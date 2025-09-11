import os
import lm_eval 
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# model_name = "/home/LLM_contrastive_loss/finetuned-sce-gemma3-4b"
# model_name = "/home/LLM_contrastive_loss/finetuned-llama-2.7b"
# model_name = "/home/LLM_contrastive_loss/finetuned-sce-llama-2.7b"
# model_name = "/home/LLM_contrastive_loss/finetuned-sce-gemma3-4b"
# model_name = "/home/llm_compression/LLaMA/gemma-3-4b-it"
# model_name = "/home/llm_compression/LLaMA/Llama-2-7b-hf"

model_name = "/home/dev/public-datasets/e.shvetsov/hugging_models/Llama-2-7b-hf" 

model = lm_eval.models.huggingface.HFLM(pretrained=model_name)
results = lm_eval.simple_evaluate(
    model=model,
    tasks=["mbpp", "mbpp_plus"],  # Replace with desired task(s) , "humaneval" "mbpp_plus"
    num_fewshot=3,
    batch_size=4096,
    device="cuda:0",
    # metric_list=[{'metric': '!function utils.pass_at_k',
    #     'aggregation': 'mean',
    #     'higher_is_better': True,
    #     'k': [10]}
    # ],
    apply_chat_template=False,
    confirm_run_unsafe_code=True
)
print(model_name)
print(results["results"])