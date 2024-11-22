from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


def load_hydra_config_from_root(
    config_path: str, config_name: str = "config"
) -> DictConfig:
    """
    Load configurations dynamically using Hydra's defaults system from a root configuration file.

    Args:
        config_path (str): The root directory containing the main config.yaml file and sub-configurations.
        config_name (str): The name of the root configuration file (default: "config").

    Returns:
        DictConfig: The merged configuration based on the defaults in the root configuration file.
    """

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    initialize(config_path=config_path, job_name="load_config")

    hydra_config = compose(config_name=config_name)
    config_dict = OmegaConf.to_object(hydra_config)
    return config_dict
