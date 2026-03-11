import torch

from cellkit.data.dataset import AnnDataDataset
from cellkit.data.reader import DataReader


class FakeReader(DataReader):
    def __init__(self, rows, obs_rows, close_count: int = 0):
        self.rows = rows
        self.obs_rows = obs_rows
        self.close_count = close_count

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.rows[0])

    @property
    def var_names(self) -> list[str]:
        return [f"feature_{index}" for index in range(len(self.rows[0]))]

    @property
    def obs_columns(self) -> list[str]:
        return [str(column) for column in self.obs_rows[0].keys()]

    @property
    def obs_names(self) -> list[str]:
        return [f"obs_{index}" for index in range(len(self.rows))]

    def read_x(self, index: int, layer: str | None = None):
        return self.rows[index]

    def read_obs(self, index: int, columns: list[str] | None = None):
        row = dict(self.obs_rows[index])
        if columns is None:
            return row
        return {column: row[column] for column in columns}

    def close(self) -> None:
        self.close_count += 1


class FakeReaderFactory:
    def __init__(self, rows, obs_rows):
        self.rows = rows
        self.obs_rows = obs_rows
        self.created_readers: list[FakeReader] = []

    def __call__(self) -> FakeReader:
        reader = FakeReader(self.rows, self.obs_rows)
        self.created_readers.append(reader)
        return reader


def test_anndata_dataset_reads_features_and_metadata():
    reader_factory = FakeReaderFactory(
        rows=[[1.0, 2.0], [3.0, 4.0]],
        obs_rows=[
            {"cell_type": "t", "target": 0},
            {"cell_type": "b", "target": 1},
        ],
    )
    dataset = AnnDataDataset(
        reader_factory,
        obs_columns=["cell_type"],
        target_column="target",
        return_index=True,
    )

    sample = dataset[1]

    assert torch.equal(sample["x"], torch.tensor([3.0, 4.0]))
    assert sample["obs"] == {"cell_type": "b"}
    assert sample["target"] == 1
    assert sample["index"] == 1
    assert len(reader_factory.created_readers) == 2
    assert dataset._reader is reader_factory.created_readers[1]
    assert reader_factory.created_readers[0].close_count == 1


def test_anndata_dataset_supports_subsetting_and_transform():
    reader_factory = FakeReaderFactory(
        rows=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        obs_rows=[{"target": 0}, {"target": 1}, {"target": 2}],
    )
    dataset = AnnDataDataset(
        reader_factory,
        indices=[2, 0],
        target_column="target",
        transform=lambda sample: {**sample, "x": sample["x"] + 1},
    )

    assert len(dataset) == 2
    sample = dataset[0]
    assert torch.equal(sample["x"], torch.tensor([6.0, 7.0]))
    assert sample["target"] == 2


def test_anndata_dataset_rejects_duplicate_target_column():
    reader_factory = FakeReaderFactory(rows=[[1.0, 2.0]], obs_rows=[{"target": 0}])

    try:
        AnnDataDataset(
            reader_factory, obs_columns=["target"], target_column="target"
        )
    except ValueError as exc:
        assert "target_column must not also appear in obs_columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_anndata_dataset_works_with_multi_worker_dataloader():
    reader_factory = FakeReaderFactory(
        rows=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        obs_rows=[
            {"target": 0},
            {"target": 1},
            {"target": 2},
            {"target": 3},
        ],
    )
    dataset = AnnDataDataset(reader_factory, target_column="target")
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2)

    batches = list(loader)

    assert len(batches) == 2
    assert torch.equal(
        batches[0]["x"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    )
    assert torch.equal(batches[1]["target"], torch.tensor([2, 3]))
