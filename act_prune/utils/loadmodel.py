from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model(model_path, seqlen):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code = True,
        torch_dtype='auto',
        device_map = 'cuda:0',
    )
    model.seqlen = seqlen

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer