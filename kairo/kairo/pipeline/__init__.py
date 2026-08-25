"""End-to-end pipelines: a config per perspective, one result shape."""
from .config import CONFIGS, InterConfig, IntraConfig, ResourceConfig
from .run import StateResult, run, run_inter_case, run_intra_case, run_resource
