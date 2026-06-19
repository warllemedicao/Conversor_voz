from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

# Configurações de Caminhos e Versão
PACKAGER_VERSION = "2026.06.19.turbo.v6"
DEFAULT_SOURCE_URL = "https://huggingface.co/buckets/warllem/Voz_Noslen"
DEFAULT_VOICE_DIR = "voices/v_minha_voz_f5_tts_ptbr"
TURBO_DURATION = 128
MEL_CHANNELS = 100
SAMPLE_RATE = 24000
FULL_PIPELINE_SMOKE_TEXT = os.environ.get(
    "F5_TTS_SMOKE_TEXT",
    "Teste rapido de validacao do pipeline completo.",
)
FULL_PIPELINE_NFE_STEP = int(os.environ.get("F5_TTS_SMOKE_NFE_STEP", "32"))

def resolve_work_root() -> Path:
    configured = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
    try:
        configured.mkdir(parents=True, exist_ok=True)
        return configured
    except OSError:
        return Path("/tmp/voz_noslen_turbo_working")

WORK_ROOT = resolve_work_root()
DOWNLOAD_DIR = WORK_ROOT / "turbo_source_snapshot"
STAGING_DIR = WORK_ROOT / "turbo_staging_area"
LOG_PATH = WORK_ROOT / "voz_noslen_turbo_packager.log"

@dataclass(frozen=True)
class TurboPaths:
    source: Path
    staging: Path
    onnx: Path
    model: Path
    reference: Path
    manifest: Path
    metadata: Path
    validation: Path

@dataclass(frozen=True)
class SourceAssets:
    voice_dir: Path
    checkpoint: Path
    vocab: Path
    reference_audio: Path
    source_manifest: Path | None = None
    source_manifest_data: dict[str, Any] | None = None

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("turbo_packager")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    return logger

LOGGER = setup_logging()

# --- Utilitários de Arquivo ---

def clean_and_make_paths() -> TurboPaths:
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    
    paths = TurboPaths(
        source=DOWNLOAD_DIR,
        staging=STAGING_DIR,
        onnx=STAGING_DIR / "onnx",
        model=STAGING_DIR / "model",
        reference=STAGING_DIR / "reference",
        manifest=STAGING_DIR / "manifest.json",
        metadata=STAGING_DIR / "metadata.json",
        validation=STAGING_DIR / "validation.json"
    )
    
    for p in [paths.onnx, paths.model, paths.reference]:
        p.mkdir(parents=True, exist_ok=True)
    return paths

def copy_readonly(src: Path, dst: Path):
    """Copia arquivos garantindo que a origem não seja alterada."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def find_source_manifest(voice_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = [
        voice_dir / "manifest.json",
        voice_dir / "model" / "manifest.json",
        voice_dir / "metadata.json",
    ]
    for candidate in candidates:
        data = read_json_if_exists(candidate)
        if data is not None:
            return candidate, data
    return None, None

def checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    step = int(digits) if digits else -1
    return (step, path.stat().st_mtime, path.name)

def select_checkpoint(model_dir: Path, source_manifest: dict[str, Any] | None) -> Path:
    manifest_candidates: list[str] = []
    if source_manifest:
        for key in ("checkpoint", "checkpoint_path", "ckpt", "model_file"):
            value = source_manifest.get(key)
            if isinstance(value, str):
                manifest_candidates.append(value)
        files = source_manifest.get("files")
        if isinstance(files, dict):
            for key in ("checkpoint", "model", "ckpt"):
                value = files.get(key)
                if isinstance(value, str):
                    manifest_candidates.append(value)

    for value in manifest_candidates:
        candidate = (model_dir.parent / value).resolve()
        if candidate.exists() and candidate.suffix == ".pt":
            return candidate
        candidate = (model_dir / value).resolve()
        if candidate.exists() and candidate.suffix == ".pt":
            return candidate

    preferred = model_dir / "model_last.pt"
    if preferred.exists():
        return preferred

    checkpoints = sorted(model_dir.glob("model_*.pt"), key=checkpoint_sort_key)
    if not checkpoints:
        checkpoints = sorted(model_dir.glob("*.pt"), key=checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"Nenhum checkpoint .pt encontrado em {model_dir}")
    return checkpoints[-1]

def discover_source_assets(source_root: Path = DOWNLOAD_DIR, voice_dir_name: str = DEFAULT_VOICE_DIR) -> SourceAssets:
    voice_dir = source_root / voice_dir_name
    if not voice_dir.exists():
        matches = sorted(source_root.glob("**/v_minha_voz_f5_tts_ptbr"))
        if matches:
            voice_dir = matches[0]
    if not voice_dir.exists():
        raise FileNotFoundError(f"Diretorio da voz nao encontrado em {source_root}/{voice_dir_name}")

    manifest_path, manifest_data = find_source_manifest(voice_dir)
    model_dir = voice_dir / "model"
    checkpoint = select_checkpoint(model_dir, manifest_data)
    vocab = model_dir / "vocab.txt"
    reference_audio = voice_dir / "data_reference" / "referencia_voz.wav"

    return SourceAssets(
        voice_dir=voice_dir,
        checkpoint=checkpoint,
        vocab=vocab,
        reference_audio=reference_audio,
        source_manifest=manifest_path,
        source_manifest_data=manifest_data,
    )

def validate_source_assets(assets: SourceAssets) -> dict[str, Any]:
    import soundfile as sf

    checks: dict[str, Any] = {
        "voice_dir": str(assets.voice_dir),
        "checkpoint": str(assets.checkpoint),
        "vocab": str(assets.vocab),
        "reference_audio": str(assets.reference_audio),
        "source_manifest": str(assets.source_manifest) if assets.source_manifest else None,
    }
    missing = [
        str(path)
        for path in (assets.checkpoint, assets.vocab, assets.reference_audio)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Arquivos obrigatorios ausentes: {missing}")

    vocab_lines = [line for line in assets.vocab.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not vocab_lines:
        raise ValueError(f"Vocabulario vazio: {assets.vocab}")

    info = sf.info(str(assets.reference_audio))
    if info.frames <= 0:
        raise ValueError(f"Audio de referencia sem frames: {assets.reference_audio}")

    checks.update(
        {
            "checkpoint_bytes": assets.checkpoint.stat().st_size,
            "vocab_entries": len(vocab_lines),
            "reference_sample_rate": info.samplerate,
            "reference_duration_sec": round(info.frames / float(info.samplerate), 3),
            "reference_channels": info.channels,
        }
    )
    if info.samplerate != SAMPLE_RATE:
        checks["reference_sample_rate_note"] = (
            f"Referencia esta em {info.samplerate} Hz; o pipeline F5-TTS/Vocos normaliza a saida para {SAMPLE_RATE} Hz."
        )
    return checks

# --- Wrapper de Exportação ---

def export_turbo_core(checkpoint_path: Path, vocab_path: Path, paths: TurboPaths, manifest_data: dict):
    import torch
    import gc
    import inspect
    from f5_tts.infer.utils_infer import load_model
    from hydra.utils import get_class

    class F5TTSTurboWrapper(torch.nn.Module):
        """Contrato Turbo: x, cond, text_ids, text_lengths, time_steps -> dx"""
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.transformer = getattr(model, "transformer", model)
            params = inspect.signature(self.transformer.forward).parameters
            self.supports_mask = "mask" in params
            self.supports_cache = "cache" in params

        def forward(self, x, cond, text_ids, text_lengths, time_steps):
            # DiT.forward espera (x, cond, text, time, ...). Passar
            # text_lengths como quinto argumento posicional faz o modelo
            # interpretá-lo como drop_audio_cond/mask e quebra audio_mask.sum(dim=1).
            length_anchor = text_lengths.to(dtype=x.dtype).reshape(1, 1, 1)
            x = x + length_anchor - length_anchor
            audio_mask = torch.ones_like(x[:, :, 0], dtype=torch.bool)
            kwargs = {"x": x, "cond": cond, "text": text_ids, "time": time_steps}
            if self.supports_mask:
                kwargs["mask"] = audio_mask
            if self.supports_cache:
                kwargs["cache"] = False
            return self.transformer(**kwargs)

    # Configuração da arquitetura (F5-TTS v1 Base)
    arch_cfg = {
        "dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512,
        "text_mask_padding": True, "conv_layers": 4, "attn_backend": "torch"
    }
    
    onnx_file = paths.onnx / "f5_tts_transformer_core.onnx"
    LOGGER.info(f"Carregando modelo para exportação: {checkpoint_path.name}")
    
    from f5_tts.model import DiT
    model = load_model(
        DiT, arch_cfg, str(checkpoint_path), 
        mel_spec_type="vocos", vocab_file=str(vocab_path), device="cpu"
    )
    model.eval()
    
    wrapped = F5TTSTurboWrapper(model).eval()
    LOGGER.info(f"Assinatura transformer.forward: {inspect.signature(wrapped.transformer.forward)}")
    
    # Exemplo de entrada para o tracer (FP32)
    example_inputs = (
        torch.randn(1, TURBO_DURATION, MEL_CHANNELS, dtype=torch.float32),       # x
        torch.randn(1, TURBO_DURATION, MEL_CHANNELS, dtype=torch.float32),       # cond
        torch.randint(0, 100, (1, 64), dtype=torch.int64),   # text_ids
        torch.tensor([64], dtype=torch.int64),               # text_lengths
        torch.tensor([0.5], dtype=torch.float32),            # time_steps
    )

    LOGGER.info("Iniciando torch.onnx.export (Turbo Contract)...")
    torch.onnx.export(
        wrapped, example_inputs, str(onnx_file),
        input_names=["x", "cond", "text_ids", "text_lengths", "time_steps"],
        output_names=["dx"],
        dynamic_axes={
            "text_ids": {1: "text_len"}
        },
        opset_version=17, do_constant_folding=True, dynamo=False
    )
    
    del model
    gc.collect()
    return onnx_file

# --- Geração de Metadados ---

def generate_metadata(paths: TurboPaths, source_info: dict):
    metadata = {
        "project": "Voz_Noslen Turbo",
        "format": "partial-core-onnx",
        "not_end_to_end": True,
        "quality_policy": "FP32 export only; no quantization, pruning, vocoder swap, or voice identity changes.",
        "contract": {
            "inputs": ["x", "cond", "text_ids", "text_lengths", "time_steps"],
            "outputs": ["dx"],
            "opset": 17,
            "dtype": "float32",
            "description": "DiT/F5-TTS core step. This graph predicts dx for one diffusion step/chunk; it does not tokenize text, run the ODE loop, run Vocos, or write WAV."
        },
        "shapes": {
            "x": [1, TURBO_DURATION, MEL_CHANNELS],
            "cond": [1, TURBO_DURATION, MEL_CHANNELS],
            "text_ids": [1, "text_len"],
            "text_lengths": [1],
            "time_steps": [1],
            "dx": [1, TURBO_DURATION, MEL_CHANNELS]
        },
        "constraints": {
            "duration": TURBO_DURATION,
            "mel_channels": MEL_CHANNELS,
            "sample_rate": SAMPLE_RATE,
            "vocoder": "Vocos",
            "reason": "Legacy TorchScript ONNX trace specializes DiT text embedding length via seq_len.max().item()."
        },
        "python_pipeline_components": [
            "reference audio preprocessing",
            "text normalization/tokenization",
            "diffusion/ODE loop orchestration",
            "Vocos vocoder",
            "WAV writing at 24000 Hz"
        ],
        "engine": "ONNX Runtime CPU for DiT core plus Python F5-TTS/Vocos pipeline"
    }
    paths.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

def generate_manifest(paths: TurboPaths, source_info: dict, runtime_files: dict):
    manifest = {
        "name": f"Voz_Noslen_Turbo_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "version": PACKAGER_VERSION,
        "backend": "lite-cpu-partial-onnx",
        "onnx_scope": "f5_tts_dit_core_only",
        "source_origin": source_info.get("repo_id", "HuggingFace Buckets"),
        "source_voice_dir": source_info.get("voice_dir", DEFAULT_VOICE_DIR),
        "source_checkpoint": source_info.get("checkpoint"),
        "sample_rate": SAMPLE_RATE,
        "vocoder": "vocos",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": runtime_files
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def validate_onnx_core(paths: TurboPaths) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort
    import numpy as np

    report: dict[str, Any] = {
        "scope": "onnx_core_only",
        "status": "pending",
        "checks": {},
        "runtime": {"provider": "CPUExecutionProvider"},
    }
    
    # 1. Check ONNX integrity
    try:
        model = onnx.load(str(paths.onnx / "f5_tts_transformer_core.onnx"))
        onnx.checker.check_model(model)
        report["checks"]["onnx_structure"] = "valid"
    except Exception as e:
        report["checks"]["onnx_structure"] = f"invalid: {str(e)}"

    # 2. ORT Smoke Test
    try:
        sess = ort.InferenceSession(str(paths.onnx / "f5_tts_transformer_core.onnx"), providers=["CPUExecutionProvider"])
        report["checks"]["onnxruntime_load"] = "success"
        report["inputs"] = [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in sess.get_inputs()
        ]
        report["outputs"] = [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in sess.get_outputs()
        ]
        
        # Teste de inferência dummy
        feeds = {
            "x": np.random.randn(1, TURBO_DURATION, MEL_CHANNELS).astype(np.float32),
            "cond": np.random.randn(1, TURBO_DURATION, MEL_CHANNELS).astype(np.float32),
            "text_ids": np.zeros((1, 16), dtype=np.int64),
            "text_lengths": np.array([16], dtype=np.int64),
            "time_steps": np.array([0.5], dtype=np.float32)
        }
        start = time.perf_counter()
        outputs = sess.run(None, feeds)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if outputs[0].shape != (1, TURBO_DURATION, MEL_CHANNELS):
            raise RuntimeError(f"dx shape inesperado: {outputs[0].shape}")
        report["runtime"]["core_inference_ms"] = round(elapsed_ms, 3)
        report["runtime"]["dx_shape"] = list(outputs[0].shape)
        report["checks"]["inference_smoke_test"] = "passed"
        report["status"] = "verified" if report["checks"].get("onnx_structure") == "valid" else "failed"
    except Exception as e:
        report["checks"]["inference_smoke_test"] = f"failed: {str(e)}"
        report["status"] = "failed"

    return report

def call_with_supported_kwargs(fn: Any, kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(fn)
    accepted = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return fn(**accepted)

def run_full_pipeline_smoke(assets: SourceAssets, output_wav: Path) -> dict[str, Any]:
    import soundfile as sf
    import torch
    from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
    from f5_tts.model import DiT

    report: dict[str, Any] = {
        "scope": "python_full_pipeline",
        "status": "pending",
        "output_wav": str(output_wav),
        "sample_rate": SAMPLE_RATE,
        "vocoder": "vocos",
        "nfe_step": FULL_PIPELINE_NFE_STEP,
        "text": FULL_PIPELINE_SMOKE_TEXT,
        "note": "This smoke test uses the original PyTorch F5-TTS pipeline. It validates WAV generation; it is intentionally separate from the ONNX core smoke test.",
    }

    arch_cfg = {
        "dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512,
        "text_mask_padding": True, "conv_layers": 4, "attn_backend": "torch"
    }
    start = time.perf_counter()
    device = "cpu"
    model = load_model(
        DiT, arch_cfg, str(assets.checkpoint),
        mel_spec_type="vocos", vocab_file=str(assets.vocab), device=device
    )
    vocoder = call_with_supported_kwargs(
        load_vocoder,
        {
            "vocoder_name": "vocos",
            "is_local": False,
            "local_path": "",
            "device": device,
        },
    )
    ref_audio, ref_text = preprocess_ref_audio_text(str(assets.reference_audio), "")

    infer_kwargs = {
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "gen_text": FULL_PIPELINE_SMOKE_TEXT,
        "model_obj": model,
        "vocoder": vocoder,
        "mel_spec_type": "vocos",
        "nfe_step": FULL_PIPELINE_NFE_STEP,
        "speed": 1.0,
    }
    result = call_with_supported_kwargs(infer_process, infer_kwargs)
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError(f"infer_process retornou formato inesperado: {type(result)!r}")
    wav, sr = result[0], result[1]
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), wav, int(sr))
    info = sf.info(str(output_wav))
    report.update(
        {
            "status": "generated",
            "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
            "written_sample_rate": info.samplerate,
            "duration_sec": round(info.frames / float(info.samplerate), 3),
            "channels": info.channels,
        }
    )
    if info.samplerate != SAMPLE_RATE:
        raise RuntimeError(f"WAV smoke test gerou sample rate {info.samplerate}, esperado {SAMPLE_RATE}")
    return report

def validate_package(paths: TurboPaths, assets: SourceAssets):
    report: dict[str, Any] = {
        "status": "pending",
        "version": PACKAGER_VERSION,
        "contract": "partial-core-onnx-plus-python-pipeline",
        "asset_checks": {},
        "onnx_core": {},
        "full_pipeline": {},
    }

    try:
        report["asset_checks"] = validate_source_assets(assets)
    except Exception as e:
        report["asset_checks"] = {"status": "failed", "error": str(e)}

    report["onnx_core"] = validate_onnx_core(paths)

    try:
        smoke_wav = paths.staging / "validation" / "full_pipeline_smoke.wav"
        report["full_pipeline"] = run_full_pipeline_smoke(assets, smoke_wav)
    except Exception as e:
        report["full_pipeline"] = {
            "scope": "python_full_pipeline",
            "status": "failed",
            "error": str(e),
            "note": "The ONNX core can still be valid, but the Kaggle flow is not complete until WAV generation passes."
        }

    report["status"] = (
        "verified"
        if report.get("onnx_core", {}).get("status") == "verified"
        and report.get("full_pipeline", {}).get("status") == "generated"
        and "error" not in report.get("asset_checks", {})
        else "failed"
    )
    paths.validation.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

# --- Orquestração Principal ---

def main():
    LOGGER.info(f"=== Voz_Noslen Turbo Packager {PACKAGER_VERSION} ===")
    
    # Simulação de localização de arquivos (No Kaggle isso viria do download)
    # Aqui assumimos que o download do bucket HF já foi feito pelo notebook
    source_root = DOWNLOAD_DIR
    paths = clean_and_make_paths()

    assets = discover_source_assets(source_root, DEFAULT_VOICE_DIR)
    source_info = {
        "repo_id": DEFAULT_SOURCE_URL,
        "voice_dir": str(assets.voice_dir.relative_to(source_root)) if assets.voice_dir.is_relative_to(source_root) else str(assets.voice_dir),
        "checkpoint": assets.checkpoint.name,
        "source_manifest": str(assets.source_manifest) if assets.source_manifest else None,
    }
    LOGGER.info(f"Voz fonte: {assets.voice_dir}")
    LOGGER.info(f"Checkpoint selecionado: {assets.checkpoint.name}")
    LOGGER.info(f"Manifesto fonte: {source_info['source_manifest'] or 'nao encontrado'}")
    validate_source_assets(assets)

    # 1. Copiar ativos fixos para o pacote
    copy_readonly(assets.vocab, paths.model / "vocab.txt")
    copy_readonly(assets.reference_audio, paths.reference / "referencia_voz.wav")
    
    # 2. Exportar ONNX
    try:
        export_turbo_core(assets.checkpoint, assets.vocab, paths, assets.source_manifest_data or {})
    except Exception as e:
        LOGGER.exception(f"Falha na exportação ONNX: {e}")
        sys.exit(1)

    # 3. Gerar Metadados e Manifest
    runtime_files = {
        "onnx": "onnx/f5_tts_transformer_core.onnx",
        "vocab": "model/vocab.txt",
        "reference": "reference/referencia_voz.wav"
    }
    generate_metadata(paths, source_info)
    generate_manifest(paths, source_info, runtime_files)

    # 4. Validar
    report = validate_package(paths, assets)
    LOGGER.info(f"Validação concluída: {report['status']}")

    # 5. Finalizar (O notebook fará o zip/upload)
    LOGGER.info(f"Pacote Turbo gerado em: {paths.staging}")

if __name__ == "__main__":
    main()
