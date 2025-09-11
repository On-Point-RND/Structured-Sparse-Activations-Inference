from datasets import load_dataset


# Load and process wikitext2 dataset
def get_wikitext2(nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    # Load test datasets
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    trainloader = None
    return trainloader, testenc