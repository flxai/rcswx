import copy
import os
import shutil

import pytest
import torch

from tests.reference import table42


@pytest.fixture
def runtime_config():
    previous_threads = torch.get_num_threads()
    config = table42.resolve_runtime_config(min(os.sched_getaffinity(0)), True)
    yield config
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize(
    ("symbol", "count", "tokens"),
    [
        ("resnet18_no_maxpool", 267, 130),
        ("resnet34_no_maxpool", 499, 242),
        ("mlpmixer_d8", 264, 125),
        ("mlpmixer_d12", 392, 185),
    ],
)
def test_supplied_models_replay_exactly_with_independent_backend_ownership(
    runtime_config, symbol, count, tokens
):
    roots, records = [], []
    for owned in (False, True):
        root, _, record = table42.model_root(symbol, owned, runtime_config)
        assert len(list(root.serialise())) == count
        assert record["alignment_tokens_including_start"] == tokens
        assert record["runtime_sha256"] == record["baseline_runtime_sha256"]
        roots.append(root)
        records.append(record)
    assert type(roots[0]) is not type(roots[1])
    assert table42.runtime_tree_record(roots[0]) == table42.runtime_tree_record(roots[1])
    if symbol.startswith("mlpmixer"):
        relative = [
            row for row in records[0]["operation_mapping"] if row["source"].startswith("linear_x")
        ]
        assert {row["runtime"]["name"] for row in relative} == {"linear(x4)", "linear(x0.25)"}
        assert {tuple(row["runtime"]["infer"]["args"]) for row in relative} == {(4,), (0.25,)}


def test_ordinary_grammar_retains_the_supplied_mixer_admission_failure(runtime_config):
    strict = table42.resolve_runtime_config(runtime_config["affinity_cpu"])
    for owned in (False, True):
        with pytest.raises(RuntimeError):
            table42.model_root("mlpmixer_d8", owned, strict)


def test_incomplete_stream_is_rejected_instead_of_randomly_completed(runtime_config):
    types = table42.backend(False)
    guard = types.limiter(copy.deepcopy(runtime_config["construction_limits"]))
    pcfg = types.pcfg(table42.grammar_for(types, "grammar", []), guard)
    sampler = types.sampler(pcfg, guard, "iterative")
    previous = table42.forbid_random_fallback(pcfg)
    inputs = copy.deepcopy(runtime_config["input_params"])
    inputs["shape"] = torch.Size(inputs["shape"])
    guard.timer.start()
    try:
        with pytest.raises(RuntimeError):
            sampler.sample(inputs, [types.grammar.sequential_module])
    finally:
        pcfg.sample = previous


def test_supplied_grouping_and_source_bytes_cannot_be_silently_replaced(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(table42.BUNDLE_ROOT, bundle)
    path = bundle / "fixtures/expanded/resnet18_no_maxpool.einspace.txt"
    path.write_text(path.read_text().replace("computation[identity]", "identity", 1))
    with pytest.raises(ValueError):
        table42.verify_assets(bundle)


@pytest.mark.parametrize("mutation", ["grammar", "limiter", "historical", "admission_name"])
def test_unresolved_or_mislabelled_runtime_configurations_are_rejected(runtime_config, mutation):
    config = copy.deepcopy(runtime_config)
    if mutation == "grammar":
        config["replay_grammar"] = "missing_grammar"
    elif mutation == "limiter":
        del config["alignment_limits"]["memory_crossover"]
    elif mutation == "historical":
        config["historical"] = True
    else:
        config["profile_id"] = "diagnostic_cpu32_v1"
    with pytest.raises(ValueError):
        table42.validate_runtime_config(config)
