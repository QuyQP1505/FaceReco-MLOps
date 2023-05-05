"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline, node

from face_rec_avenger.nodes import (
    extract_features,
    train_model,
    recognize_faces,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                extract_features,
                ["train_images", "parameters"],
                "features",
                name="extract_features",
            ),
            node(
                train_model,
                ["features", "parameters"],
                "svm_model",
                name="train_model",
            ),
            node(
                recognize_faces,
                ["test_images", "svm_model", "features"],
                "face_predictions",
                name="recognize_faces",
            ),
        ]
    )
    
    
# def register_pipelines() -> Dict[str, Pipeline]:
#     """Register the project's pipelines.

#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """
#     pipelines = find_pipelines()
#     pipelines["__default__"] = sum(pipelines.values())
#     return pipelines


def register_pipelines():
    return {"recognition_pipeline": create_pipeline()}