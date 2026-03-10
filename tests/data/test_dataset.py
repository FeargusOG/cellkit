import torch

from cellkit.data.dataset import AnnDataDataset
from cellkit.data.reader import DataReader


class FakeReader(DataReader):
    def __init__(self, rows, obs_rows, clone_count: int = 0, close_count: int = 0):
        self.rows = rows
        self.obs_rows = obs_rows
        self.clone_count = clone_count
        self.close_count = close_count

    def clone(self) -> "FakeReader":
        self.clone_count += 1
        return FakeReader(self.rows, self.obs_rows)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.rows[0])

    def read_x(self, index: int, layer: str | None = None):
        return self.rows[index]

    def read_obs(self, index: int, columns: list[str] | None = None):
        row = dict(self.obs_rows[index])
        if columns is None:
            return row
        return {column: row[column] for column in columns}

    def close(self) -> None:
        self.close_count += 1


def test_anndata_dataset_reads_features_and_metadata():
    reader = FakeReader(
        rows=[[1.0, 2.0], [3.0, 4.0]],
        obs_rows=[
            {"cell_type": "t", "target": 0},
            {"cell_type": "b", "target": 1},
        ],
    )
    dataset = AnnDataDataset(
        reader,
        obs_columns=["cell_type"],
        target_column="target",
        return_index=True,
    )

    sample = dataset[1]

    assert torch.equal(sample["x"], torch.tensor([3.0, 4.0]))
    assert sample["obs"] == {"cell_type": "b"}
    assert sample["target"] == 1
    assert sample["index"] == 1
    assert reader.clone_count == 1


def test_anndata_dataset_supports_subsetting_and_transform():
    reader = FakeReader(
        rows=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        obs_rows=[{"target": 0}, {"target": 1}, {"target": 2}],
    )
    dataset = AnnDataDataset(
        reader,
        indices=[2, 0],
        target_column="target",
        transform=lambda sample: {**sample, "x": sample["x"] + 1},
    )

    assert len(dataset) == 2
    sample = dataset[0]
    assert torch.equal(sample["x"], torch.tensor([6.0, 7.0]))
    assert sample["target"] == 2


def test_anndata_dataset_rejects_duplicate_target_column():
    reader = FakeReader(rows=[[1.0, 2.0]], obs_rows=[{"target": 0}])

    try:
        AnnDataDataset(reader, obs_columns=["target"], target_column="target")
    except ValueError as exc:
        assert "target_column must not also appear in obs_columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_anndata_dataset_works_with_multi_worker_dataloader():
    reader = FakeReader(
        rows=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        obs_rows=[
            {"target": 0},
            {"target": 1},
            {"target": 2},
            {"target": 3},
        ],
    )
    dataset = AnnDataDataset(reader, target_column="target")
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2)

    batches = list(loader)

    assert len(batches) == 2
    assert torch.equal(
        batches[0]["x"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    )
    assert torch.equal(batches[1]["target"], torch.tensor([2, 3]))
