"""Evaluate config schema"""
from pydantic import BaseModel, Literal

class EvalRecapConfig(BaseModel):
    """docstring to complete"""
    metrics: list[Literal["metric1", "metric2", "metric3"]] = ["metric1", "metric2"]
    graphs: list[Literal["graph1", "graph2"]] = ["graph1", "graph2"]