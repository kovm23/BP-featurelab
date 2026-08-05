"""Persistence tests for MachineLearningPipeline checkpoint save/load."""
import glob
import os

import pandas as pd

from pipeline import MachineLearningPipeline


def test_save_load_roundtrip(tmp_path):
    folder = str(tmp_path)
    pipe = MachineLearningPipeline(checkpoint_folder=folder)
    pipe.feature_spec = {"brightness": [0, 255], "scene": ["day", "night"]}
    pipe.target_variable = "mood"
    pipe.target_mode = "classification"
    pipe.rules = ["IF brightness > 100 THEN day"]
    pipe.training_X = pd.DataFrame({"brightness": [10, 200], "scene": ["night", "day"]})
    pipe.save_state()

    restored = MachineLearningPipeline(checkpoint_folder=folder)
    assert restored.load_state() is True
    assert restored.target_variable == "mood"
    assert restored.target_mode == "classification"
    assert restored.feature_spec == {"brightness": [0, 255], "scene": ["day", "night"]}
    assert restored.rules == ["IF brightness > 100 THEN day"]
    assert restored.training_X is not None
    pd.testing.assert_frame_equal(restored.training_X, pipe.training_X)


def test_save_leaves_no_temp_files(tmp_path):
    folder = str(tmp_path)
    pipe = MachineLearningPipeline(checkpoint_folder=folder)
    pipe.feature_spec = {"f": [0, 1]}
    pipe.target_variable = "t"
    pipe.training_X = pd.DataFrame({"f": [0.1, 0.9]})
    pipe.save_state()

    # The atomic writer must clean up after itself: no ".tmp-" residue.
    leftovers = glob.glob(os.path.join(folder, ".tmp-*"))
    assert leftovers == []
    assert os.path.exists(os.path.join(folder, "pipeline_state.json"))
    assert os.path.exists(os.path.join(folder, "training_X.csv"))


def test_save_removes_stale_files_when_data_cleared(tmp_path):
    folder = str(tmp_path)
    pipe = MachineLearningPipeline(checkpoint_folder=folder)
    pipe.training_X = pd.DataFrame({"f": [1, 2]})
    pipe.save_state()
    assert os.path.exists(os.path.join(folder, "training_X.csv"))

    # Clearing the data and saving again must delete the now-stale CSV.
    pipe.training_X = None
    pipe.save_state()
    assert not os.path.exists(os.path.join(folder, "training_X.csv"))
