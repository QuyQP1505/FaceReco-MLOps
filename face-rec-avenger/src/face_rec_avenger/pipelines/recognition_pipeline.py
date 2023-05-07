from kedro.pipeline import Pipeline, node
from face_rec_avenger.nodes import (
    extract_features,
    classify,
    predict,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                extract_features.extract_features,
                ["test_images", "parameters"],
                "features",
                name="extract_features",
            ),
            # node(
            #     classify.classify,
            #     ["features", "parameters"],
            #     "svm_model",
            #     name="train_model",
            # ),
            node(
                predict.predict,
                ["svm_model", "features"],
                "face_predictions",
                name="recognize_faces",
            ),
        ]
    )
