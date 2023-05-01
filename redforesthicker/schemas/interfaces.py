"""Schemas definition for input and output"""
from pydantic import BaseModel, Literal, StrictBool

class DataFormatConfig(BaseModel):
    """Data format configuration model"""
    extension: Literal["json", "csv", "txt"] = "csv"
    layout: Literal["top-down", "left-right"] = "top-down"
    header: StrictBool = True

class InputConfig(BaseModel):
    """Input configuration interface"""
    input_path: str = "./input"
    targets: Literal["merged", "separated"]
    variable_path: str | None = None
    target_path: str | None = None
    data_format: DataFormatConfig


class OutputConfig(BaseModel):
    """Output config interface"""
    output_path: str = "./output"
    keep_data: StrictBool = False


class SavingConfig(BaseModel):
    """Saving config interface"""
    mode: Literal["pickle", "skops", "onnx", "pmml"] = "onnx"


class LoadingConfig(BaseModel):
    """Loading config interface"""
    mode = Literal["pickle", "skops", "onnx", "pmml"] = "onnx"