"""Train configuration schemas"""
from pathlib import Path
from pydantic import BaseModel, Literal

from redforesthicker.schemas.interfaces import InputConfig, OutputConfig
from redforesthicker.schemas.evaluate import EvalRecapConfig
from redforesthicker.schemas.models import ModelConfig

class PredictConfig(BaseModel):
    input: InputConfig
    data_preprocess = None #TODO: implementation of a data pipeline
    output: OutputConfig
    model: ModelConfig


class PredictTask(BaseModel):
    name: str = "predict"
    input_path: Path
    output_path: Path
    config: PredictConfig
    eval_recap: EvalRecapConfig | None