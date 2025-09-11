import logging
import os
import torch
from abc import ABC, abstractmethod
from typing import Tuple

from utils.loadmodel import get_model
from utils.getwiki import get_wikitext2

import uuid
import lm_eval
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

class BaseRunner(ABC):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.run_id = uuid.uuid4()
        logging.info(f"Run ID: {self.run_id}")


    def load_model_tokenizer(self):
        path_to_model = self.config["model"]["path"]
        seqlen = self.config["model"]["seqlen"]
        self.model, self.tokenizer = get_model(path_to_model, seqlen)

    def load_data(self, dataset_name):
        """Load dataset for pruning and validation """
        if dataset_name == "wikitext2":
            trainloader, testenc = get_wikitext2(
                seqlen=self.model.seqlen,
                tokenizer=self.tokenizer
            )

        return trainloader, testenc

    def replace_mlp_blocks(self):
        """Insert into model modified mlp blocks with original weights """
        raise NotImplementedError

    @torch.no_grad()
    def measure_ppl(self, testenc, bs=1):
        """Measure quality of sparsified model"""

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        device = self.model.device

        # Get input IDs
        testenc = testenc.input_ids

        # Calculate number of samples
        nsamples = testenc.numel() // self.model.seqlen

        # List to store negative log likelihoods
        nlls = []
        print(f"nsamples {nsamples}")

        # Loop through each batch
        start.record()
        for i in range(0,nsamples,bs):
            if i % 50 == 0:
                print(f"sample {i}")

            # Calculate end index
            j = min(i+bs, nsamples)

            # Prepare inputs and move to device
            inputs = testenc[:,(i * self.model.seqlen):(j * self.model.seqlen)].to(device)
            inputs = inputs.reshape(j-i, self.model.seqlen)

            # Forward pass through the model
            lm_logits = self.model(inputs).logits

            # Shift logits and labels for next token prediction
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # Compute loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

            # Calculate negative log likelihood
            neg_log_likelihood = loss.float() * self.model.seqlen * (j-i)

            # Append to list of negative log likelihoods
            nlls.append(neg_log_likelihood)

        # Compute perplexity
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * self.model.seqlen))
        end.record()
        # Empty CUDA cache to save memory
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return ppl.item(), start.elapsed_time(end) / 1000

    def run_lm_eval(self):
        config = self.config["benchmarks"]["harness"]
        model = lm_eval.models.huggingface.HFLM(pretrained=self.model)
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=config["tasks"],  # Replace with desired task(s)
            num_fewshot=config["num_fewshot"],
            batch_size=config["batch_size"],
            apply_chat_template=config["apply_chat_template"],
            confirm_run_unsafe_code=True,
            device="cuda:0"
        )
        return results["results"]

    def setup_environment(self):
        os.environ["CUDA_DEVICE_ORDER"] = self.config["env"]["CUDA_DEVICE_ORDER"]
        os.environ["OMP_NUM_THREADS"] = self.config["env"]["OMP_NUM_THREADS"]
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config["env"]["CUDA_VISIBLE_DEVICES"]

    @abstractmethod
    def run(self):
        """Run method to be implemented in derived classes."""
        raise NotImplementedError