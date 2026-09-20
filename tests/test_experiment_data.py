"""Check endpoint semantics and that held-out data cannot change preprocessing."""
import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas", reason="Optional experiment data dependencies")

from benchmarks import experiment_data as data


def test_survival_target_keeps_censoring_and_time():
    y = data.survival_target([2.5, 8], [0, 1])
    assert y.dtype.names == ("event", "time")
    np.testing.assert_array_equal(y["event"], [False, True])
    np.testing.assert_array_equal(y["time"], [2.5, 8])


@pytest.mark.parametrize("time,event", [([0], [1]), ([-1], [1]), ([np.nan], [1]),
                                            ([1], [2]), ([1], [np.nan]), ([1, 2], [1])])
def test_invalid_survival_targets_rejected(time, event):
    with pytest.raises(ValueError):
        data.survival_target(time, event)


def test_training_statistics_are_frozen_for_held_out_rows():
    train = np.array([[1, np.nan, 7, 1], [3, np.nan, 7, np.nan], [5, np.nan, 7, 5]])
    prep = data.TrainingPreprocessor.fit(train, block_size=2)
    np.testing.assert_array_equal(prep.keep, [True, False, False, True])
    transformed = prep.transform(train, block_size=1)
    np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(transformed.std(axis=0), 1, atol=1e-7)
    means = prep.mean.copy()
    held_out = prep.transform(np.array([[101, 5, 80, np.nan]]))
    np.testing.assert_array_equal(prep.mean, means)
    np.testing.assert_allclose(held_out[0], [(101 - 3) / np.sqrt(8 / 3), 0], rtol=1e-6)


def test_alternate_endpoint_mask_keeps_features_and_ids_aligned(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA", tmp_path)
    folder = tmp_path / "example"
    folder.mkdir()
    X = np.array([[11, 12], [21, 22], [31, 32]], dtype=np.float32)
    np.save(folder / "X.npy", X)
    pd.DataFrame({"patient_id": ["p1", "p2", "p3"], "PFI_time": [3, np.nan, 7],
                  "PFI_event": [0, np.nan, 1]}).to_csv(folder / "samples.tsv", sep="\t", index=False)
    pd.DataFrame({"feature_id": ["a", "b"]}).to_csv(folder / "features.tsv", sep="\t", index=False)
    (folder / "metadata.json").write_text(json.dumps({"kind": "survival", "endpoints": ["PFI"],
                                                     "n_samples": 3, "events": 2}))
    result = data.load_survival("example", endpoint="PFI")
    np.testing.assert_array_equal(result.X, X[[0, 2]])
    assert result.samples.patient_id.tolist() == ["p1", "p3"]
    assert result.metadata["n_samples"] == 2
    assert result.metadata["events"] == 1
    assert result.metadata["prepared_n_samples"] == 3


def test_unusable_training_matrix_rejected():
    with pytest.raises(ValueError, match="No nonconstant"):
        data.TrainingPreprocessor.fit(np.array([[np.nan, 1], [np.nan, 1]]))
    with pytest.raises(ValueError, match="Infinite"):
        data.TrainingPreprocessor.fit(np.array([[np.inf], [1]]))
