from transformers import AutoModel, AutoModelForCausalLM
from transformers.pipelines import PIPELINE_REGISTRY

from .rep_control_pipeline import RepControlPipeline


def repe_pipeline_registry():

    PIPELINE_REGISTRY.register_pipeline(
        "ctg-control",
        pipeline_class=RepControlPipeline,
        pt_model=AutoModelForCausalLM,
    )


