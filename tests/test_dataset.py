"""Dataset abstraction: the PanNuke spec and the extension path (custom dataset)."""

import numpy as np

from vlm_medseg.data import PANNUKE_SPEC, available_datasets, get_dataset, register_dataset
from vlm_medseg.data.base import BaseDataset, DatasetSpec
from vlm_medseg.types import Sample


def test_pannuke_spec_and_registry():
    assert PANNUKE_SPEC.num_classes == 5
    assert len(PANNUKE_SPEC.group_names) == 19
    assert PANNUKE_SPEC.class_name(0) == "Neoplastic"
    assert "pannuke" in available_datasets()


def test_sam3_concept_prompts_and_fallback():
    # PanNuke defines short SAM3 concepts, distinct from the long VLM prompts.
    assert PANNUKE_SPEC.concept_prompt(0) == "tumor cell nucleus"
    assert PANNUKE_SPEC.generic_concept_prompt == "cell nucleus"
    # A dataset that omits concept_prompts falls back to its class prompts/names.
    assert ToyDataset.spec.concept_prompt(0) == "an a-thing"
    assert ToyDataset.spec.generic_concept_prompt == "object"


class ToyDataset(BaseDataset):
    spec = DatasetSpec(
        name="toy",
        class_names=["a", "b"],
        class_prompts={0: "an a-thing", 1: "a b-thing"},
        group_names=["g0", "g1"],
        class_colors={0: (255, 0, 0), 1: (0, 0, 255)},
    )

    def __init__(self, n: int = 10):
        self.n = n

    def __len__(self):
        return self.n

    def decode(self, idx: int) -> Sample:
        m = np.zeros((10, 10), np.int32)
        m[0:5, 0:5] = 1
        return Sample(image=np.zeros((10, 10, 3), np.uint8), inst_map=m,
                      inst_classes={1: idx % 2}, sample_id=f"toy-{idx}",
                      group=idx % 2, dataset="toy")

    def group_ids(self):
        return [i % 2 for i in range(self.n)]


def test_custom_dataset_registers_and_runs():
    register_dataset("toy", ToyDataset)
    ds = get_dataset("toy", n=12)
    assert ds.spec.num_classes == 2
    idx = ds.stratified_indices(6, seed=0)
    assert len(idx) == 6 and idx == sorted(idx)
    # both groups represented
    assert {ds.group_ids()[i] for i in idx} == {0, 1}
    s = ds.decode(1)
    assert s.num_instances == 1 and s.dataset == "toy"


def test_stratify_random_when_ungrouped():
    class Flat(ToyDataset):
        def group_ids(self):
            return [-1] * self.n

    ds = Flat(n=20)
    idx = ds.stratified_indices(5, seed=1)
    assert len(idx) == 5 and len(set(idx)) == 5
