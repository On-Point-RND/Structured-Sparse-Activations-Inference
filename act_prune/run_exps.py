import os
import logging
import warnings
import yaml
from utils.basic import seed_everything, load_config 
from act_prune_runner import ActPruneRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
warnings.filterwarnings("ignore")

def main() -> None:
    config_dir = "./configs"
    base_config_path = os.path.join(config_dir, "config.yaml")
    config = load_config(base_config_path, config_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = config['env']["CUDA_VISIBLE_DEVICES"]
    logging.info("Configuration:\n%s", yaml.dump(config))
    seed_everything(config["env"]["SEED"])
    logging.info(f"Fixing seed: {config['env']['SEED']}")
    runner = ActPruneRunner(config)

    runner.run()


if __name__ == "__main__":
    main()