import json
import logging
from pathlib import Path
from base_runner_class import BaseRunner
from modelling import layer_swap

from training.global_train import run_train
from training.sequential_train import sequential_parameter_training
from training.grad_accumulation import compute_grads

class ActPruneRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
     
    def replace_linear_layers(self):
        """Insert into model modified linear layers with original weights """
        logging.info("Replace Linear layers...")
        
        model = getattr(self.model,'model',self.model)

        layer_swap.swap_linear_inplace(model,
                                       self.config["pruning"]["sparsity_type"],
                                       self.config["pruning"]["transformation_type"],
                                       self.config["pruning"].get("sparsity_ratio", None),
                                       self.config["pruning"].get("prune_n", None),
                                       self.config["pruning"].get("prune_m", None),
                                       self.config["pruning"]["target_modules"],
                                       logging)

        logging.info("Modified model...")
        logging.info(self.model)        

    def replace_mlp_blocks(self):
        """Insert into model modified mlp blocks with original weights """
        logging.info("Replace MLP blocks...")
        layer_swap.swap_mlp_inplace(self.model.model,
                                    self.model.model.layers[0].mlp, 
                                    self.config["pruning"]["sparsity_type"],
                                    logging)
        
        logging.info("Modified model...")
        logging.info(self.model)

    def replace_attn_blocks(self):
        """Insert into model modified self attn blocks with original weights """

        logging.info("Replace SelfAttn blocks...")
        
        layer_swap.swap_attention_inplace(self.model.model,
                                    self.model.model.layers[0].self_attn,
                                    self.config["pruning"]["sparsity_type"],
                                    logging)

        logging.info("Modified model...")
        logging.info(self.model)

    def run(self):
        """Execute the pruning pipeline."""
        self.load_model_tokenizer()

        log_dir = Path(self.config["paths"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        res_file = log_dir / f"{self.run_id}.json"

        res = {}
        res["config"] = self.config
        sparsity_type = self.config["pruning"]["sparsity_type"]
        if sparsity_type == "None":
            logging.info("No sparsity applied, using original model.")
        else:
            self.replace_linear_layers()

        if self.config["pruning"]["transformation_type"] == "learnable":
            trainloader, testloader = self.load_data("wikitext2")
            if self.config["finetuning"]["type"] == "global":
                self.model = run_train(self.model, self.tokenizer, self.config)
                for name, param in self.model.named_parameters():
                    param.requires_grad = False
            elif self.config["finetuning"]["type"] == "by_layers":
                sequential_parameter_training(self.config, self.model, trainloader)

        if self.config["pruning"]["sparsity_type"] == "semi-structured_act_grad_acc":
            compute_grads(self.model, self.tokenizer, self.config)
                

        benchmarks = self.config["benchmarks"]

        if benchmarks["ppl_wikitext2"]["run_ppl"]:
            trainloader, testloader = self.load_data("wikitext2")
            ppl, time = self.measure_ppl(testloader, bs=benchmarks["ppl_wikitext2"]["batch_size"])

            logging.info(f'wikitext2: {ppl}, computation time: {time}')
            res["wikitext ppl"] = ppl
            res["wikitext time"] = time


        if benchmarks["harness"]["run_lm_eval"]:
            results = self.run_lm_eval()
            logging.info(results)
            res["lm_eval"] = results

        with res_file.open("w") as f:
            json.dump(res, f, indent=2)
            logging.info("Results saved to %s", res_file)