import logging
from base_runner_class import BaseRunner
from modelling import layer_swap

class ActPruneRunner(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
     
    def replace_linear_layers(self):
        """Insert into model modified linear layers with original weights """
        logging.info("Replace Linear layers...")
        layer_swap.swap_linear_inplace(self.model.model,
                                       self.config["pruning"]["sparsity_type"],
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

        if self.config["pruning"]["module"] == "layers":
            print('layers!')
            self.replace_linear_layers()
        elif self.config["pruning"]["module"] == "mlp_blocks":
            print('mlp_blocks!')
            self.replace_mlp_blocks()
        elif self.config["pruning"]["module"] == "attn_blocks":
            print('attn_blocks!')
            self.replace_attn_blocks()

        benchmarks = self.config["benchmarks"]
        if benchmarks["ppl_wikitext2"]["run_ppl"]:
            _, testloader = self.load_data("wikitext2")
            # Evaluate ppl in no grad context to avoid updating the model
            ppl, time = self.measure_ppl(testloader)
            logging.info(f'wikitext2: {ppl}, computation time: {time}')
        
        if benchmarks["harness"]["run_lm_eval"]:
            results = self.run_lm_eval()
            logging.info(results)
