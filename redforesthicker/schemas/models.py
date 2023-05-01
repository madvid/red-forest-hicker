"""Model configuration schemas"""
from pydantic import BaseModel, Literal, Extra, validator, root_validator
from redforesthicker.schemas.interfaces import LoadingConfig, SavingConfig

AVAILABLE_MODELS = ["decision_tree_classifier",
                    "decision_tree_regressor",
                    "random_forest",
                    "isolation_forest",
                    "gradient_boosting",
                    "xgboost"]

class DecisionTreeBaseParamters(BaseModel):
    """Decision Tree Base Parameters"""
    splitter='best',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.0


class DecisionTreeClassifierParameters(DecisionTreeBaseParamters, Extra.forbid):
    """docstring to complete"""
    criterion: Literal["gini", "entropy", "log_loss"] = "gini"
    

class DecisionTreeRegressorParameters(DecisionTreeBaseParamters, Extra.forbid):
    """docstring to complete"""
    criterion: Literal["squared_error", "absolute_error", "friedman_mse", "poisson"] = 'squared_error'


class ForestBaseParameters(BaseModel):
    """Forest Base Parameters"""
    max_samples=None
    max_features='sqrt'
    estimators=100
    bootstrap=True
    n_jobs=None
    random_state=None
    verbose=0
    warm_start=False

class RandomForestParameters(ForestBaseParameters, Extra.forbid):
    """docstring to complete"""
    tree_type: Literal["classifier", "regressor"]
    # depending of the type a subset of criterion is allowed
    criterion: Literal["gini", "entropy", "log_loss", "squared_error", "absolute_error", "friedman_mse", "poisson"]
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    oob_score=False,
    class_weight=None,
    ccp_alpha=0.0,

    @validator
    def chek_tree_type_criterion_matching():
        #TODO:
        tree_type = ...
        criterion = ...
        if tree_type == "classifier":
            if not criterion in ["gini", "entropy", "log_loss"]:
                raise ValueError(
                    f"""
                    
                    """
                )
        else:
            if not criterion in ["squared_error", "absolute_error", "friedman_mse", "poisson"]:
                raise ValueError(
                    f"""
                    
                    """
                )
        return 


class IsolationForestParameters(ForestBaseParameters, Extra.forbid):
    """docstring to complete"""
    contamination='auto'


class GradientBoostingClassifierParameters(BaseModel, Extra.forbid):
    """docstring to complete"""
    loss='log_loss',
    learning_rate=0.1,
    n_estimators=100,
    subsample=1.0,
    criterion='friedman_mse',
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=3,
    min_impurity_decrease=0.0,
    init=None,
    random_state=None,
    max_features=None,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False,
    validation_fraction=0.1,
    n_iter_no_change=None,
    tol=0.0001,
    ccp_alpha=0.0


class GradientBoostingRegressorParamters(BaseModel, Extra.forbid):
    """doctstring to complete"""
    oss='squared_error',
    learning_rate=0.1,
    n_estimators=100,
    subsample=1.0,
    criterion='friedman_mse',
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=3,
    min_impurity_decrease=0.0,
    init=None,
    random_state=None,
    max_features=None,
    alpha=0.9,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False,
    validation_fraction=0.1,
    n_iter_no_change=None,
    tol=0.0001,
    ccp_alpha=0.0

class XGBoostParameters(BaseModel, Extra.forbid):
    """docstring to complete"""
    pass

MODELS_PARAMETERS = DecisionTreeClassifierParameters | DecisionTreeRegressorParameters | RandomForestParameters | IsolationForestParameters |GradientBoostingParameters | XGBoostParameters

class ModelConfig(BaseModel):
    name: Literal[AVAILABLE_MODELS] = "decision_tree_classifiers"
    loading: LoadingConfig
    saving: SavingConfig
    parameters: MODELS_PARAMETERS