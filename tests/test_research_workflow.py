import pathlib

import pytest

from run_experiment import require_mode_allowed, research_probe_next_step


def test_research_probe_never_recommends_automatic_scaling():
    message = research_probe_next_step().lower()

    assert "accept/reject/inconclusive" in message
    assert "kanonik" in message
    assert "tam veri kümesiyle devam" not in message
    assert "parametrelerini artır" not in message


@pytest.mark.parametrize(
    "mode",
    ["overfit", "micro-pilot", "local-train", "modal-pilot", "all"],
)
def test_legacy_training_modes_are_disabled_during_clean_start(mode):
    candidate_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "configs"
        / "research_candidate.yaml"
    )

    with pytest.raises(RuntimeError, match="Eski eğitim modu.*devre dışı"):
        require_mode_allowed(mode=mode, candidate_path=candidate_path)


@pytest.mark.parametrize("mode", ["local-test", "summary", "download-hf"])
def test_non_training_modes_remain_available(mode):
    candidate_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "configs"
        / "research_candidate.yaml"
    )

    require_mode_allowed(mode=mode, candidate_path=candidate_path)


def test_legacy_modal_entrypoint_uses_the_same_training_guard():
    cloud_runner = (
        pathlib.Path(__file__).resolve().parent.parent
        / "src"
        / "modal_runner"
        / "cloud_train.py"
    )

    source = cloud_runner.read_text(encoding="utf-8")

    assert 'require_mode_allowed(mode="modal-pilot")' in source
