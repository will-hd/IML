import json
import copy
from typing import TypedDict
"""
Example json
{
    "experiment": {
        "meta": {
            "name": "DefaultExperiment"
        },
        "model": {
            "n_layers": 151
            "x_size": 1,
            "y_size": 1,
            "r_size": 64,
            "z_size": 64,
            "h_size_dec": 64,
            "h_size_enc_lat": 64,
            "h_size_enc_det": 64,
            "N_h_layers_dec": 3,
            "N_xy_to_si_layers": 2,
            "N_sc_to_qz_layers": 1,
            "N_h_layers_enc_det": 6,
            "use_r": false
        },
        "training": {
            "optimiser": {
                "algorithm": "Adam",
                "LR": 4e-4
            },
            "batch_size": 8,
            "iterations": 10000
        },
        "data": {
            "max_num_context": 100,
            "num_points": 400,
            "l1_scale": "0.6",
            "sigma_scale": 1.0,
            "random_kernel_parameters": true,
        }
    }
}
"""
class MetaConfigType(TypedDict):
    name: str

class ModelConfigType(TypedDict):
    n_layers: int
    x_size: int
    y_size: int
    r_size: int
    z_size: int
    h_size_dec: int
    h_size_enc_lat: int
    h_size_enc_det: int
    N_h_layers_dec: int
    N_h_layers_enc_lat_phi: int
    N_h_layers_enc_lat_rho: int
    N_h_layers_enc_det: int
    use_r: bool

class OptimiserConfigType(TypedDict):
    algorithm: str
    LR: float

class TrainingConfigType(TypedDict):
    optimiser: OptimiserConfigType
    batch_size: int
    iterations: int

class DataConfigType(TypedDict):
    max_num_context: int
    num_points: int
    l1_scale: float
    sigma_scale: float
    random_kernel_parameters: bool

class ExperimentConfigType(TypedDict):
    meta: MetaConfigType
    model: ModelConfigType
    training: TrainingConfigType
    data: DataConfigType

class ConfigType(TypedDict):
    experiment: ExperimentConfigType


class Config():  
    """Simple dict wrapper that adds a thin API allowing for slash-based retrieval of
    nested elements, e.g. cfg.get("meta/dataset_name")
    """
    def __init__(self, config_path, default_path=None):
        with open(config_path) as cf_file:
            cfg = json.load(cf_file)
            cfg = ConfigType(cfg)
        
        if default_path is not None:
            with open(default_path) as def_cf_file:
                default_cfg = json.load(def_cf_file)
                
            Config.merge_dictionaries_recursively(default_cfg, cfg)
            
            cfg = default_cfg
        
        self._data = cfg

    def get(self, path=None, default=None):
        # We need to deep-copy self._data to avoid over-writing its data
        sub_dict = copy.deepcopy(self._data)

        if path is None:
            return sub_dict

        path_items = path.split("/")
        try:
            for path_item in path_items:
                sub_dict = sub_dict[path_item]

            return sub_dict
        except (KeyError, TypeError):
            return default

    @staticmethod
    def merge_dictionaries_recursively(default, override):
        """Recursively merges two dictionaries.
        Values from 'override' will overwrite those from 'default'.
        """
        for key, value in override.items():
            if isinstance(value, dict):
                if key not in default:
                    default[key] = {}
                Config.merge_dictionaries_recursively(default[key], value)
            else:
                default[key] = value

