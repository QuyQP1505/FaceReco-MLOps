"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline, node

from face_rec_avenger.pipelines import (
    recognition_pipeline as rcp
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    recognition_pipeline = rcp.create_pipeline()

    return {
        "__default__": recognition_pipeline,
        "recognition_pipeline": recognition_pipeline,
    }