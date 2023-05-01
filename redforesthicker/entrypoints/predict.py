"""Prediction entrypoint."""
import argparse
from pathlib import Path
from redforesthicker.schemas.predict import PredictConfig

def parser_fnc():
    """ parser function of the prediction entrypoints"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predict_config",
        "-c",
        type=Path,
        required=True,
        help="""
        Predict configuration path yaml file (see 'Prediction/Configuration' section documentation for more details).
        """,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        required=True,
        help="""
        model file (see 'Prediction/Loading a model' section documentation for more details).
        """
    )
    parser.add_argument(
        "--overwrite",
        "-f",
        required=False,
        default=False,
        type=bool,
        action="strore_true",
        help="""
        In the case output directory exists, if overwrite is set to True, the output directory contain will be overwrite
        """
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="""
        Input directory where data is expected to be.
        Input layout depends on what it is precised in the predict configuration file.
        """
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="""
        Output directory where to store the output of the prediction.
        Either a existing or non existing directory.
        """
    )

    args = parser.parse_args()

    return args


def main():
    """main function of prediction entrypoint"""
    args = parser_fnc()
    predict_config = PredictConfig.parse_file(args.predict_config)
    


if "__name__" == "__main":
    main()