import hydra
from experiments.experiment import AmplitudeExperiment
import sys

@hydra.main(config_path="config", config_name="amplitudes", version_base=None)
def main(cfg):
    if cfg.exp_type == "amplitudes":
        exp = AmplitudeExperiment(cfg)

    exp()

if __name__ == "__main__":
    main()

