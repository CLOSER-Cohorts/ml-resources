# importing the module
import pytest

import numpy as np
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


class MockModel:
    classes_ = np.array(["topic_1", "topic_2"])

    def predict_proba(self, X):
        # Return probabilities for two classes
        return np.array([
            [0.9, 0.1],
            [0.2, 0.8]
        ])


@pytest.fixture
def client():

    with patch("mlflow.sklearn.load_model") as mock_model_load, \
         patch("src.ml_resources.apply_pipeline") as mock_pipeline, \
         patch("projects.am1_project.src.utility.convert_df_to_ndarray") as mock_convert:

        mock_model_load.return_value = MockModel()

        # Fake transformed embeddings output
        import pandas as pd

        mock_pipeline.return_value = pd.DataFrame({
            "summary_embeddings": [[1,2],[3,4]],
            "category_embeddings": [[5,6],[7,8]],
            "item_type": [1,0],
            "agency_id": [5,3],
            "has_categories": [1,1]
        })

        mock_convert.return_value = np.array([
            [1,2,5,6,1,5,1],
            [3,4,7,8,0,3,1]
        ])

        from projects.am1_project.api.main import app

        yield TestClient(app)