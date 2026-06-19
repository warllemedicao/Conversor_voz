from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

# Configurações de Caminhos e Versão
PACKAGER_VERSION = "2026.06.19.turbo.v5"
DEFAULT_SOURCE_URL = "https://huggingface.co/buckets/warllem/Voz_Noslen"
DEFAULT_VOICE_DIR = "voices/v_minha_voz_f5_tts_ptbr"
TURBO_DURATION = 128

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
        torch.randn(1, TURBO_DURATION, 100, dtype=torch.float32),       # x
        torch.randn(1, TURBO_DURATION, 100, dtype=torch.float32),       # cond
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
        "contract": {
            "inputs": ["x", "cond", "text_ids", "text_lengths", "time_steps"],
            "outputs": ["dx"],
            "opset": 17,
            "dtype": "float32"
        },
        "shapes": {
            "x": [1, TURBO_DURATION, 100],
            "cond": [1, TURBO_DURATION, 100],
            "text_ids": [1, "text_len"],
            "text_lengths": [1],
            "time_steps": [1],
            "dx": [1, TURBO_DURATION, 100]
        },
        "constraints": {
            "duration": TURBO_DURATION,
            "reason": "Legacy TorchScript ONNX trace specializes DiT text embedding length via seq_len.max().item()."
        },
        "engine": "ONNX Runtime CPU / Cloud Run Turbo Backend"
    }
    paths.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

def generate_manifest(paths: TurboPaths, source_info: dict, runtime_files: dict):
    manifest = {
        "name": f"Voz_Noslen_Turbo_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "version": PACKAGER_VERSION,
        "backend": "turbo-onnx",
        "source_origin": source_info.get("repo_id", "HuggingFace Buckets"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": runtime_files
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def validate_package(paths: TurboPaths):
    import onnx
    import onnxruntime as ort
    import numpy as np

    report = {"status": "pending", "checks": {}}
    
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
        
        # Teste de inferência dummy
        feeds = {
            "x": np.random.randn(1, TURBO_DURATION, 100).astype(np.float32),
            "cond": np.random.randn(1, TURBO_DURATION, 100).astype(np.float32),
            "text_ids": np.zeros((1, 16), dtype=np.int64),
            "text_lengths": np.array([16], dtype=np.int64),
            "time_steps": np.array([0.5], dtype=np.float32)
        }
        outputs = sess.run(None, feeds)
        if outputs[0].shape != (1, TURBO_DURATION, 100):
            raise RuntimeError(f"dx shape inesperado: {outputs[0].shape}")
        report["checks"]["inference_smoke_test"] = "passed"
        report["status"] = "verified"
    except Exception as e:
        report["checks"]["inference_smoke_test"] = f"failed: {str(e)}"
        report["status"] = "failed"

    paths.validation.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

# --- Orquestração Principal ---

def main():
    LOGGER.info(f"=== Voz_Noslen Turbo Packager {PACKAGER_VERSION} ===")
    
    # Simulação de localização de arquivos (No Kaggle isso viria do download)
    # Aqui assumimos que o download do bucket HF já foi feito pelo notebook
    source_root = DOWNLOAD_DIR
    paths = clean_and_make_paths()

    # Localização dos arquivos fonte (Read-Only)
    # Procurando em subpastas caso seja um bucket HF
    v_dir = source_root / DEFAULT_VOICE_DIR
    ckpt = next(v_dir.glob("model/model_*.pt"))
    vocab = v_dir / "model/vocab.txt"
    ref = v_dir / "data_reference/referencia_voz.wav"

    # 1. Copiar ativos fixos para o pacote
    copy_readonly(vocab, paths.model / "vocab.txt")
    copy_readonly(ref, paths.reference / "referencia_voz.wav")
    
    # 2. Exportar ONNX
    try:
        export_turbo_core(ckpt, vocab, paths, {})
    except Exception as e:
        LOGGER.exception(f"Falha na exportação ONNX: {e}")
        sys.exit(1)

    # 3. Gerar Metadados e Manifest
    runtime_files = {
        "onnx": "onnx/f5_tts_transformer_core.onnx",
        "vocab": "model/vocab.txt",
        "reference": "reference/referencia_voz.wav"
    }
    generate_metadata(paths, {})
    generate_manifest(paths, {}, runtime_files)

    # 4. Validar
    report = validate_package(paths)
    LOGGER.info(f"Validação concluída: {report['status']}")

    # 5. Finalizar (O notebook fará o zip/upload)
    LOGGER.info(f"Pacote Turbo gerado em: {paths.staging}")

if __name__ == "__main__":
    main()
