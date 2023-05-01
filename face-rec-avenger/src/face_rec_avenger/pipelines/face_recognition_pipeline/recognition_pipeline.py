from kedro.pipeline import Pipeline, node
from .nodes import (
    extract_features,
    classify,
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
                classify,
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
