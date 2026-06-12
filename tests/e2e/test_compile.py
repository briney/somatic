"""End-to-end tests for torch.compile + gradient (activation) checkpointing.

CPU tests use the ``aot_eager`` backend, which exercises the AOT autograd code path
(the one that breaks if a graph output regresses to a plain Python float, or if the
SAC policy / checkpoint higher-order op fails to trace) without needing inductor, a
C++ compiler, or a GPU. The CUDA-gated tests run the real (inductor) compile path
through the Trainer.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from somatic.data import create_dataloader
from somatic.model import SomaticConfig, SomaticForMaskedLM
from somatic.training.trainer import Trainer, TrainingConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_dynamo():
    """Clear the dynamo/compile cache around a test so graphs don't leak between tests."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.fixture
def restore_optimize_ddp():
    """Save and restore the process-global ``optimize_ddp`` dynamo flag."""
    import torch._dynamo

    prev = torch._dynamo.config.optimize_ddp
    yield
    torch._dynamo.config.optimize_ddp = prev


@pytest.fixture
def training_data(tmp_path):
    """Small paired heavy/light CSV for trainer-level tests."""
    data = {
        "heavy_chain": [
            "EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMH",
            "QVQLQQSGAELARPGASVKMSCKASGYTFTRYTMH",
            "EVQLVQSGAEVKKPGESLKISCKGSGYSFTSYWIGWV",
            "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAIS",
        ]
        * 5,
        "light_chain": [
            "DIQMTQSPSSVSASVGDRVTITCRASQGISSWLA",
            "DIVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGY",
            "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSY",
            "DIQMTQSPSSLSASVGDRVTITCRASQSISSYL",
        ]
        * 5,
    }
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


def _make_config(**overrides) -> SomaticConfig:
    base = dict(
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    base.update(overrides)
    return SomaticConfig(**base)


def _make_batch(seq_len: int, batch_size: int = 2) -> dict[str, torch.Tensor]:
    """A masked batch: amino-acid tokens, CLS/EOS, two chains, a few masked positions."""
    input_ids = torch.randint(4, 28, (batch_size, seq_len))
    input_ids[:, 0] = 0  # CLS
    input_ids[:, -1] = 2  # EOS
    token_type_ids = torch.cat(
        [
            torch.zeros(batch_size, seq_len // 2),
            torch.ones(batch_size, seq_len - seq_len // 2),
        ],
        dim=1,
    ).long()
    attention_mask = torch.ones(batch_size, seq_len)
    # HF-style labels: -100 everywhere except a few interior positions to score.
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    masked = slice(seq_len // 4, seq_len // 4 + 3)
    labels[:, masked] = input_ids[:, masked]
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _step(model, batch) -> torch.Tensor:
    """Forward (in-model masked-CE loss) + backward; return the loss."""
    outputs = model(
        input_ids=batch["input_ids"],
        token_type_ids=batch["token_type_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    loss = outputs.loss
    loss.backward()
    return loss


# ---------------------------------------------------------------------------
# CPU: tracing / dynamic-shape / checkpointing composition (aot_eager backend)
# ---------------------------------------------------------------------------


def test_compile_aot_eager_forward_backward(reset_dynamo) -> None:
    """torch.compile(dynamic=True) must trace forward+backward without error."""
    model = SomaticForMaskedLM(_make_config()).train()
    compiled = torch.compile(model, dynamic=True, backend="aot_eager")
    loss = _step(compiled, _make_batch(40))
    assert torch.isfinite(loss)


def test_compile_dynamic_no_recompile_across_seq_lengths(reset_dynamo) -> None:
    """dynamic=True: varying per-batch sequence length must not trigger recompiles."""
    import torch._dynamo

    model = SomaticForMaskedLM(_make_config()).train()
    compiled = torch.compile(model, dynamic=True, backend="aot_eager")

    # Warm up two lengths to establish the dynamic graph.
    for seq_len in (40, 56):
        _step(compiled, _make_batch(seq_len))

    # Now any new length (still <= max_seq_len) must reuse the compiled graph.
    prev = torch._dynamo.config.error_on_recompile
    torch._dynamo.config.error_on_recompile = True
    try:
        for seq_len in (48, 60):
            loss = _step(compiled, _make_batch(seq_len))
            assert torch.isfinite(loss)
    finally:
        torch._dynamo.config.error_on_recompile = prev


@pytest.mark.parametrize("mode", ["full", "selective"])
def test_compile_with_checkpointing_aot_eager(reset_dynamo, mode) -> None:
    """torch.compile must trace a model using full / selective activation checkpointing.

    Selective inserts a create_selective_checkpoint_contexts context_fn into
    torch.utils.checkpoint; this exercises that the module-level SAC policy composes
    with AOT autograd and that no graph output regresses to a plain Python float.
    """
    config = _make_config(gradient_checkpointing=True, gradient_checkpointing_mode=mode)
    model = SomaticForMaskedLM(config).train()
    assert all(b.gradient_checkpointing for b in model.somatic.encoder.layers)
    compiled = torch.compile(model, dynamic=True, backend="aot_eager")
    loss = _step(compiled, _make_batch(40))
    assert torch.isfinite(loss)


def test_compiled_state_dict_has_no_orig_mod(reset_dynamo) -> None:
    """The raw model keeps clean state_dict keys; only the compile wrapper prefixes them."""
    model = SomaticForMaskedLM(_make_config()).train()
    compiled = torch.compile(model, dynamic=True, backend="aot_eager")
    _step(compiled, _make_batch(40))

    # The underlying module's state_dict is unaffected by compilation.
    assert not any("_orig_mod" in k for k in model.state_dict())
    # The OptimizedModule wrapper *does* prefix keys — this is exactly why the trainer
    # unwraps with keep_torch_compile=False before checkpointing.
    assert any("_orig_mod" in k for k in compiled.state_dict())


def test_compiled_model_save_pretrained_roundtrip(reset_dynamo, tmp_path) -> None:
    """save_pretrained on the raw model + from_pretrained round-trips after compilation."""
    model = SomaticForMaskedLM(_make_config()).train()
    compiled = torch.compile(model, dynamic=True, backend="aot_eager")
    _step(compiled, _make_batch(40))

    path = tmp_path / "model"  # HuggingFace directory format
    model.save_pretrained(str(path))  # raw model -> clean keys
    reloaded = SomaticForMaskedLM.from_pretrained(str(path))

    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), reloaded.state_dict().items(), strict=True
    ):
        assert k1 == k2
        assert torch.equal(v1, v2)


# ---------------------------------------------------------------------------
# CPU: DDPOptimizer gating wiring in the Trainer
# ---------------------------------------------------------------------------


def _build_trainer(training_data, *, compile: bool, gc_mode: str) -> Trainer:
    model = SomaticForMaskedLM(
        _make_config(
            max_position_embeddings=128,
            gradient_checkpointing=True,
            gradient_checkpointing_mode=gc_mode,
        )
    )
    loader = create_dataloader(
        data_path=training_data,
        batch_size=4,
        max_length=128,
        shuffle=False,
        num_workers=0,
    )
    cfg = TrainingConfig(max_steps=1, compile=compile, mixed_precision="no")
    return Trainer(config=cfg, model=model, train_dataloader=loader)


def test_compile_selective_disables_ddp_optimizer(
    training_data, reset_dynamo, restore_optimize_ddp
) -> None:
    """compile=True + selective: the trainer sets optimize_ddp=False.

    torch.compile wrapping is lazy, so Trainer.__init__ flips the flag without a real
    compilation — no train() call is needed.
    """
    import torch._dynamo

    _build_trainer(training_data, compile=True, gc_mode="selective")
    assert torch._dynamo.config.optimize_ddp is False


def test_compile_full_leaves_ddp_optimizer_default(
    training_data, reset_dynamo, restore_optimize_ddp
) -> None:
    """compile=True + full: optimize_ddp is left at its default (comm/compute overlap kept)."""
    import torch._dynamo

    default = torch._dynamo.config.optimize_ddp
    _build_trainer(training_data, compile=True, gc_mode="full")
    assert torch._dynamo.config.optimize_ddp == default


# ---------------------------------------------------------------------------
# CUDA: real (inductor) compile path through the Trainer
# ---------------------------------------------------------------------------

_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="real torch.compile path needs CUDA"
)


@pytest.mark.slow
@_REQUIRES_CUDA
def test_compiled_trainer_steps_are_finite(training_data, reset_dynamo) -> None:
    """A few real compiled training steps produce finite losses."""
    trainer = _build_trainer(training_data, compile=True, gc_mode="selective")
    trainer.model.train()
    for i, batch in enumerate(trainer.train_dataloader):
        loss, _ = trainer.training_step(batch)
        trainer.accelerator.backward(loss)
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        assert torch.isfinite(loss)
        if i >= 2:
            break


@pytest.mark.slow
@_REQUIRES_CUDA
def test_compiled_trainer_checkpoint_roundtrip(training_data, reset_dynamo, tmp_path) -> None:
    """After compiled training, the unwrapped model saves/loads with clean keys."""
    trainer = _build_trainer(training_data, compile=True, gc_mode="full")
    trainer.model.train()
    for i, batch in enumerate(trainer.train_dataloader):
        loss, _ = trainer.training_step(batch)
        trainer.accelerator.backward(loss)
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        if i >= 1:
            break

    unwrapped = trainer.accelerator.unwrap_model(trainer.model, keep_torch_compile=False)
    assert not any("_orig_mod" in k for k in unwrapped.state_dict())
    path = tmp_path / "model"  # HuggingFace directory format
    unwrapped.save_pretrained(str(path))
    reloaded = SomaticForMaskedLM.from_pretrained(str(path))
    assert reloaded.config.hidden_size == unwrapped.config.hidden_size
