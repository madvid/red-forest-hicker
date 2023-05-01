"""Train configuration schemas"""
from pydantic import BaseModel
from pydantic import Literal

from redforesthicker.schemas.interfaces import InputConfig, OutputConfig

TreeConfigs = DecisionTreeRegressorConfig | DecisionTreeClassifierConfig | GradientBosstingConfig | RandomForestConfig | IsolationForestConfig

class DecisionTreeRegressorConfig(BaseModel):
    """Configuration for Decision Tree Regressor Training"""
    pass


class DecisionTreeClassifierConfig(BaseModel):
    """Configuration for Decision Tree Classifier Training"""
    pass


class GradientBosstingConfig(BaseModel):
    """Configuration for Gradient Boosting Training"""
    pass


class RandomForestConfig(BaseModel):
    """Configuration for Random Forest Training"""
    pass


class IsolationForestConfig(BaseModel):
    """Configuration for Isolation Forest Training"""
    pass



class TrainConfig(BaseModel):
    method: Literal["decision_tree_regressor", "decision_tree_classifier", "gradient_boosting", "radom_forest", "isolation_forest"]
    config: TreeConfigs
    input: InputConfig
    output: OutputConfig