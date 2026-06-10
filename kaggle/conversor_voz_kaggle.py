from __future__ import annotations

import json
import logging
import math
import os
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


HF_REPO_ID = "warllem/Super_voz"
HF_REVISION = "main"
PREFERRED_VOICE_PACKAGE = "v_minha_voz_f5_tts_ptbr"
MODEL_ROOT = Path(os.environ.get("SUPER_VOZ_MODEL_ROOT", "/kaggle/working/Super_voz"))
OUTPUT_DIR = Path(os.environ.get("SUPER_VOZ_OUTPUT_DIR", "/kaggle/working"))
LOG_PATH = Path(os.environ.get("SUPER_VOZ_LOG_PATH", "/kaggle/working/super_voz_kaggle.log"))
TEST_TEXT = "Boa noite Warllem, sua voz esta pronta."

DEFAULT_F5TTS_V1_BASE_CONFIG: dict[str, Any] = {
    "backbone": "DiT",
    "arch": {
        "dim": 1024,
        "depth": 22,
        "heads": 16,
        "ff_mult": 2,
        "text_dim": 512,
        "text_mask_padding": True,
        "qk_norm": None,
        "conv_layers": 4,
        "pe_attn_head": None,
        "attn_backend": "torch",
        "attn_mask_enabled": False,
        "checkpoint_activations": False,
    },
    "mel_spec": {
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
        "win_length": 1024,
        "n_fft": 1024,
        "mel_spec_type": "vocos",
    },
    "tokenizer": "char",
    "vocoder": "vocos",
}

PORTUGUESE_REQUIRED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZáàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ .,!?;:-'\"")


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int | None = None
    lfs_size: int | None = None
    oid: str | None = None
    lfs_oid: str | None = None

    @property
    def expected_size(self) -> int | None:
        return self.lfs_size or self.size


@dataclass
class CheckpointChoice:
    remote_path: str
    reason: str
    expected_size: int | None = None


@dataclass
class VoicePackage:
    repo_id: str
    revision: str
    root: Path
    voice_dir: str
    manifest_path: Path
    manifest: dict[str, Any]
    checkpoint: CheckpointChoice
    vocab_remote_path: str
    base_vocab_remote_path: str | None
    reference_audio_remote_path: str
    reference_text_remote_path: str | None
    base_library_dir: str | None
    setting_remote_path: str | None
    config: dict[str, Any]
    remote_files: dict[str, RemoteFile]
    local_checkpoint_path: Path | None = None
    local_vocab_path: Path | None = None
    local_base_vocab_path: Path | None = None
    local_reference_audio_path: Path | None = None
    local_reference_text_path: Path | None = None
    local_setting_path: Path | None = None


@dataclass
class DiagnosticReport:
    rows: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, item: str, ok: bool, detail: str = "") -> None:
        self.rows.append((item, "OK" if ok else "FALHA", detail))

    @property
    def ready(self) -> bool:
        blocking = {
            "Manifesto",
            "Checkpoint principal",
            "Checkpoint legivel",
            "Arquitetura identificada",
            "Vocab encontrado",
            "Vocab compativel",
            "Referencia de audio",
            "Vocoder",
        }
        return all(status == "OK" for item, status, _ in self.rows if item in blocking)

    def render(self) -> str:
        lines = [f"{'Item':<32} {'Status':<8} Detalhe"]
        for item, status, detail in self.rows:
            lines.append(f"{item:<32} {status:<8} {detail}")
        lines.append(f"{'Inferencia pronta':<32} {'SIM' if self.ready else 'NAO':<8}")
        return "\n".join(lines)


def setup_logging(log_path: Path = LOG_PATH) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("super_voz_kaggle")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


LOGGER = setup_logging()


def log_exception_context(exc: BaseException, repo_id: str, revision: str, token: str | None) -> None:
    LOGGER.error("Falha Hugging Face sem fallback silencioso.")
    LOGGER.error("Tipo da excecao: %s", type(exc).__name__)
    LOGGER.error("Mensagem: %s", exc)
    LOGGER.error("Repo ID: %s", repo_id)
    LOGGER.error("Revisao: %s", revision)
    LOGGER.error("Token presente: %s", "sim" if token else "nao")
    LOGGER.error("Traceback completo:\n%s", "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def get_kaggle_secret(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret(name)
        return value or None
    except Exception:
        return None


def get_hf_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name) or get_kaggle_secret(name)
        if value:
            return value
    return None


def file_size_text(size: int | None) -> str:
    if size is None:
        return "desconhecido"
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    return f"{value:.2f} {unit}"


def list_remote_files(
    repo_id: str = HF_REPO_ID,
    revision: str = HF_REVISION,
    token: str | None = None,
) -> dict[str, RemoteFile]:
    from huggingface_hub import HfApi

    token = token or get_hf_token()
    api = HfApi(token=token)
    try:
        tree = list(api.list_repo_tree(repo_id=repo_id, repo_type="model", revision=revision, recursive=True, expand=True))
    except Exception as exc:
        log_exception_context(exc, repo_id, revision, token)
        raise

    remote_files: dict[str, RemoteFile] = {}
    for item in tree:
        path = getattr(item, "path", None)
        if not path or item.__class__.__name__ == "RepoFolder":
            continue
        lfs = getattr(item, "lfs", None) or {}
        lfs_size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
        lfs_oid = lfs.get("oid") if isinstance(lfs, dict) else (getattr(lfs, "oid", None) or getattr(lfs, "sha256", None))
        remote_files[path] = RemoteFile(
            path=path,
            size=getattr(item, "size", None),
            lfs_size=lfs_size,
            oid=getattr(item, "oid", None) or getattr(item, "blob_id", None),
            lfs_oid=lfs_oid,
        )

    LOGGER.info("Arquivos encontrados no Hugging Face (%s @ %s):", repo_id, revision)
    for file in sorted(remote_files.values(), key=lambda item: item.path):
        LOGGER.info("- %s (%s)", file.path, file_size_text(file.expected_size))
    return remote_files


def download_remote_file(
    remote_path: str,
    output_dir: Path,
    repo_id: str = HF_REPO_ID,
    revision: str = HF_REVISION,
    token: str | None = None,
    expected_size: int | None = None,
) -> Path:
    from huggingface_hub import hf_hub_download

    token = token or get_hf_token()
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Baixando %s", remote_path)
    try:
        local_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                filename=remote_path,
                local_dir=str(output_dir),
                token=token,
            )
        )
    except TypeError:
        local_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                filename=remote_path,
                local_dir=str(output_dir),
                token=token,
            )
        )
    except Exception as exc:
        log_exception_context(exc, repo_id, revision, token)
        raise

    validate_downloaded_file(local_path, expected_size)
    return local_path


def validate_downloaded_file(path: Path, expected_size: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo baixado nao encontrado: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError(f"Arquivo vazio: {path}")
    if expected_size is not None and size != expected_size:
        raise RuntimeError(f"Tamanho invalido em {path}: baixado={size}, esperado={expected_size}")
    LOGGER.info("Arquivo local: %s (%s)", path, file_size_text(size))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_manifest_remote_path(manifest: dict[str, Any], manifest_path: str, value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("voices/") or value.startswith("libraries/"):
        return value
    remote_dir = manifest.get("huggingface_remote_dir") or str(Path(manifest_path).parent).replace("\\", "/")
    return f"{remote_dir.rstrip('/')}/{value.lstrip('/')}"


def choose_manifest(remote_files: dict[str, RemoteFile], root: Path, repo_id: str, revision: str, token: str | None) -> tuple[str, Path, dict[str, Any]]:
    manifest_paths = sorted(path for path in remote_files if re.fullmatch(r"voices/[^/]+/manifest\.json", path))
    if not manifest_paths:
        raise RuntimeError("Nenhum manifest.json F5-TTS encontrado em voices/*/manifest.json.")

    candidates: list[tuple[int, str, Path, dict[str, Any]]] = []
    for remote_path in manifest_paths:
        local_path = download_remote_file(remote_path, root, repo_id, revision, token, remote_files[remote_path].expected_size)
        manifest = load_json(local_path)
        score = 0
        if manifest.get("architecture") == "F5-TTS":
            score += 100
        if manifest.get("package_name") == PREFERRED_VOICE_PACKAGE:
            score += 50
        if manifest.get("huggingface_remote_dir", "").endswith(PREFERRED_VOICE_PACKAGE):
            score += 30
        if remote_path.startswith(f"voices/{PREFERRED_VOICE_PACKAGE}/"):
            score += 20
        if resolve_manifest_remote_path(manifest, remote_path, manifest.get("voice_checkpoint")) in remote_files:
            score += 10
        candidates.append((score, remote_path, local_path, manifest))

    score, remote_path, local_path, manifest = sorted(candidates, reverse=True)[0]
    if manifest.get("architecture") != "F5-TTS":
        raise RuntimeError(f"Manifesto escolhido nao e F5-TTS: {remote_path}")
    LOGGER.info("Manifesto escolhido: %s (score=%s)", remote_path, score)
    return remote_path, local_path, manifest


def choose_checkpoint(manifest: dict[str, Any], manifest_remote_path: str, remote_files: dict[str, RemoteFile]) -> CheckpointChoice:
    policy = (
        ("voice_checkpoint", "checkpoint recomendado pelo manifest.json"),
        ("inference_checkpoint", "checkpoint final de inferencia do manifest.json"),
        ("final_checkpoint", "checkpoint final do manifest.json"),
        ("latest_checkpoint", "checkpoint mais recente do manifest.json"),
    )
    for key, reason in policy:
        remote_path = resolve_manifest_remote_path(manifest, manifest_remote_path, manifest.get(key))
        if remote_path and remote_path in remote_files and remote_path.endswith((".pt", ".safetensors")):
            return CheckpointChoice(remote_path, reason, remote_files[remote_path].expected_size)

    voice_dir = manifest.get("huggingface_remote_dir") or str(Path(manifest_remote_path).parent).replace("\\", "/")
    fallback_names = ("model/model_last.safetensors", "model/model_last.pt", "model/latest_checkpoint.pt")
    for name in fallback_names:
        remote_path = f"{voice_dir.rstrip('/')}/{name}"
        if remote_path in remote_files:
            return CheckpointChoice(remote_path, f"fallback validado existente: {name}", remote_files[remote_path].expected_size)

    raise RuntimeError("Nenhum checkpoint F5-TTS valido encontrado pelo manifesto ou fallbacks F5.")


def infer_reference_text_path(reference_audio_remote_path: str, remote_files: dict[str, RemoteFile]) -> str | None:
    audio = Path(reference_audio_remote_path)
    candidates = [
        str(audio.with_suffix(".txt")).replace("\\", "/"),
        str(audio.parent / "referencia_voz.txt").replace("\\", "/"),
        str(audio.parent / "reference.txt").replace("\\", "/"),
    ]
    for candidate in candidates:
        if candidate in remote_files:
            return candidate
    return None


def get_f5_config(manifest: dict[str, Any], setting: dict[str, Any] | None = None) -> dict[str, Any]:
    exp_name = manifest.get("exp_name") or (setting or {}).get("exp_name")
    if exp_name != "F5TTS_v1_Base":
        raise RuntimeError(
            "O repositorio nao contem configuracao completa para uma arquitetura diferente de F5TTS_v1_Base. "
            f"exp_name encontrado: {exp_name!r}"
        )
    config = json.loads(json.dumps(DEFAULT_F5TTS_V1_BASE_CONFIG))
    config["tokenizer"] = manifest.get("tokenizer") or (setting or {}).get("tokenizer_type") or "char"
    if config["tokenizer"] != "char":
        raise RuntimeError(f"Tokenizer nao suportado para esta voz: {config['tokenizer']}")
    return config


def build_voice_package(
    repo_id: str = HF_REPO_ID,
    revision: str = HF_REVISION,
    root: Path = MODEL_ROOT,
    token: str | None = None,
    download_checkpoint: bool = True,
) -> VoicePackage:
    token = token or get_hf_token()
    remote_files = list_remote_files(repo_id, revision, token)
    manifest_remote_path, local_manifest_path, manifest = choose_manifest(remote_files, root, repo_id, revision, token)
    voice_dir = manifest.get("huggingface_remote_dir") or str(Path(manifest_remote_path).parent).replace("\\", "/")

    checkpoint = choose_checkpoint(manifest, manifest_remote_path, remote_files)
    vocab_remote_path = f"{voice_dir.rstrip('/')}/model/vocab.txt"
    if vocab_remote_path not in remote_files:
        raise RuntimeError(f"Vocabulario da voz nao encontrado: {vocab_remote_path}")

    base_library = manifest.get("base_library") or {}
    base_library_dir = base_library.get("huggingface_remote_dir")
    base_vocab_remote_path = f"{base_library_dir}/vocab.txt" if base_library_dir else None
    if base_vocab_remote_path and base_vocab_remote_path not in remote_files:
        LOGGER.warning("Vocabulario da biblioteca-base declarado, mas ausente: %s", base_vocab_remote_path)
        base_vocab_remote_path = None

    reference_audio_remote_path = f"{voice_dir.rstrip('/')}/data_reference/referencia_voz.wav"
    if reference_audio_remote_path not in remote_files:
        raise RuntimeError(f"Audio de referencia nao encontrado: {reference_audio_remote_path}")
    reference_text_remote_path = infer_reference_text_path(reference_audio_remote_path, remote_files)

    setting_remote_path = f"{base_library_dir}/setting.json" if base_library_dir else None
    local_setting_path = None
    setting = None
    if setting_remote_path and setting_remote_path in remote_files:
        local_setting_path = download_remote_file(setting_remote_path, root, repo_id, revision, token, remote_files[setting_remote_path].expected_size)
        setting = load_json(local_setting_path)

    config = get_f5_config(manifest, setting)
    package = VoicePackage(
        repo_id=repo_id,
        revision=revision,
        root=root,
        voice_dir=voice_dir,
        manifest_path=local_manifest_path,
        manifest=manifest,
        checkpoint=checkpoint,
        vocab_remote_path=vocab_remote_path,
        base_vocab_remote_path=base_vocab_remote_path,
        reference_audio_remote_path=reference_audio_remote_path,
        reference_text_remote_path=reference_text_remote_path,
        base_library_dir=base_library_dir,
        setting_remote_path=setting_remote_path if setting_remote_path in remote_files else None,
        config=config,
        remote_files=remote_files,
        local_setting_path=local_setting_path,
    )
    download_required_artifacts(package, token=token, download_checkpoint=download_checkpoint)
    return package


def download_required_artifacts(package: VoicePackage, token: str | None = None, download_checkpoint: bool = True) -> VoicePackage:
    token = token or get_hf_token()
    package.local_vocab_path = download_remote_file(
        package.vocab_remote_path,
        package.root,
        package.repo_id,
        package.revision,
        token,
        package.remote_files[package.vocab_remote_path].expected_size,
    )
    if package.base_vocab_remote_path:
        package.local_base_vocab_path = download_remote_file(
            package.base_vocab_remote_path,
            package.root,
            package.repo_id,
            package.revision,
            token,
            package.remote_files[package.base_vocab_remote_path].expected_size,
        )
    package.local_reference_audio_path = download_remote_file(
        package.reference_audio_remote_path,
        package.root,
        package.repo_id,
        package.revision,
        token,
        package.remote_files[package.reference_audio_remote_path].expected_size,
    )
    if package.reference_text_remote_path:
        package.local_reference_text_path = download_remote_file(
            package.reference_text_remote_path,
            package.root,
            package.repo_id,
            package.revision,
            token,
            package.remote_files[package.reference_text_remote_path].expected_size,
        )
    else:
        LOGGER.warning("Texto exato da referencia nao existe no pacote da voz; F5-TTS tentara transcrever com ASR.")

    if download_checkpoint:
        package.local_checkpoint_path = download_remote_file(
            package.checkpoint.remote_path,
            package.root,
            package.repo_id,
            package.revision,
            token,
            package.checkpoint.expected_size,
        )
    else:
        package.local_checkpoint_path = package.root / package.checkpoint.remote_path
    return package


def read_vocab(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [line.rstrip("\n\r") for line in text.splitlines()]


def validate_vocab(vocab_path: Path, base_vocab_path: Path | None = None) -> dict[str, Any]:
    tokens = read_vocab(vocab_path)
    non_empty = [token for token in tokens if token != ""]
    duplicates = sorted({token for token in non_empty if non_empty.count(token) > 1})
    chars = set("".join(non_empty))
    missing_pt = sorted(char for char in PORTUGUESE_REQUIRED_CHARS if char not in chars)
    report: dict[str, Any] = {
        "path": str(vocab_path),
        "tokens": len(tokens),
        "empty_lines": len(tokens) - len(non_empty),
        "duplicates": duplicates,
        "missing_portuguese_chars": missing_pt,
        "utf8": True,
    }
    if base_vocab_path and base_vocab_path.exists():
        base_tokens = read_vocab(base_vocab_path)
        report["base_path"] = str(base_vocab_path)
        report["base_tokens"] = len(base_tokens)
        report["same_as_base"] = tokens == base_tokens
        report["only_voice"] = sorted(set(tokens) - set(base_tokens))[:50]
        report["only_base"] = sorted(set(base_tokens) - set(tokens))[:50]
    LOGGER.info("Relatorio vocabulario: %s", json.dumps(report, ensure_ascii=False, indent=2))
    if not non_empty:
        raise RuntimeError(f"Vocabulario vazio: {vocab_path}")
    if duplicates:
        raise RuntimeError(f"Vocabulario contem tokens duplicados: {duplicates[:10]}")
    return report


def safe_torch_load(path: Path, map_location: str = "cpu") -> Any:
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def tensor_shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(part) for part in shape)


def inspect_checkpoint(checkpoint_path: Path, vocab_path: Path | None = None) -> dict[str, Any]:
    suffix = checkpoint_path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(str(checkpoint_path), device="cpu")
    else:
        checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")

    info: dict[str, Any] = {
        "path": str(checkpoint_path),
        "object_type": type(checkpoint).__name__,
        "top_level_keys": [],
        "has_ema": False,
        "has_model_state": False,
        "has_optimizer": False,
        "has_scheduler": False,
        "has_scaler": False,
        "step": None,
        "epoch": None,
        "selected_weight_key": None,
        "text_embedding_shape": None,
        "vocab_tokens": None,
        "vocab_compatible": None,
    }
    if isinstance(checkpoint, dict):
        info["top_level_keys"] = sorted(str(key) for key in checkpoint.keys())[:100]
        info["has_ema"] = any(key in checkpoint for key in ("ema_model_state_dict", "ema_model"))
        info["has_model_state"] = any(key in checkpoint for key in ("model_state_dict", "state_dict", "model"))
        info["has_optimizer"] = any("optim" in str(key).lower() for key in checkpoint)
        info["has_scheduler"] = any("sched" in str(key).lower() for key in checkpoint)
        info["has_scaler"] = any("scaler" in str(key).lower() for key in checkpoint)
        info["step"] = checkpoint.get("step") or checkpoint.get("global_step") or checkpoint.get("update")
        info["epoch"] = checkpoint.get("epoch")
        state = select_inference_state_dict(checkpoint)
        info["selected_weight_key"] = state[0]
        for key, value in state[1].items():
            if key.endswith("text_embed.weight") or "text_embed" in key and key.endswith(".weight"):
                info["text_embedding_shape"] = tensor_shape(value)
                break
    else:
        raise RuntimeError(f"Checkpoint possui tipo inesperado para inferencia: {type(checkpoint).__name__}")

    if vocab_path:
        tokens = read_vocab(vocab_path)
        info["vocab_tokens"] = len(tokens)
        shape = info["text_embedding_shape"]
        if shape:
            info["vocab_compatible"] = shape[0] in (len(tokens), len(tokens) + 1, len(tokens) + 2)

    LOGGER.info("Inspecao do checkpoint: %s", json.dumps(info, ensure_ascii=False, default=str, indent=2))
    return info


def select_inference_state_dict(checkpoint: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    candidates = (
        "ema_model_state_dict",
        "ema_model",
        "model_state_dict",
        "state_dict",
        "model",
    )
    for key in candidates:
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            if key.startswith("ema"):
                cleaned = {str(k).replace("ema_model.", ""): v for k, v in value.items() if k not in ("initted", "step")}
                return key, cleaned
            return key, value
    if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
        return "root_state_dict", checkpoint
    raise RuntimeError("Nao encontrei pesos de inferencia em state_dict/model_state_dict/ema_model_state_dict.")


def validate_reference_audio(audio_path: Path, expected_sample_rate: int = 24000) -> dict[str, Any]:
    import soundfile as sf

    info = sf.info(str(audio_path))
    data, sr = sf.read(str(audio_path), always_2d=True)
    duration = len(data) / sr if sr else 0.0
    mono = data.mean(axis=1)
    peak = float(abs(mono).max()) if len(mono) else 0.0
    rms = float(math.sqrt(float((mono * mono).mean()))) if len(mono) else 0.0
    report = {
        "path": str(audio_path),
        "format": info.format,
        "sample_rate": sr,
        "channels": data.shape[1],
        "duration": duration,
        "peak": peak,
        "rms": rms,
        "converted_path": None,
    }
    if duration <= 0 or peak <= 0 or rms <= 1e-5:
        raise RuntimeError(f"Audio de referencia invalido ou silencioso: {report}")
    if sr != expected_sample_rate or data.shape[1] != 1:
        converted = audio_path.with_name(f"{audio_path.stem}_{expected_sample_rate}hz_mono.wav")
        try:
            import librosa

            mono_24k, _ = librosa.load(str(audio_path), sr=expected_sample_rate, mono=True)
        except Exception:
            mono_24k = mono
        sf.write(str(converted), mono_24k, expected_sample_rate)
        report["converted_path"] = str(converted)
    LOGGER.info("Referencia de audio: %s", json.dumps(report, ensure_ascii=False, indent=2))
    return report


def read_reference_text(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def detect_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_speech(
    text: str,
    output_path: str,
    checkpoint_path: str,
    vocab_path: str,
    ref_audio_path: str,
    ref_text: str,
    device: str = "auto",
    model_name: str = "F5TTS_v1_Base",
    model_config: dict[str, Any] | None = None,
) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("Texto vazio para sintese.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import soundfile as sf
    import torch
    from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
    from hydra.utils import get_class

    selected_device = detect_device(device)
    config = model_config or DEFAULT_F5TTS_V1_BASE_CONFIG
    if model_name != "F5TTS_v1_Base":
        raise RuntimeError(f"Modelo F5-TTS nao suportado por este pacote: {model_name}")

    model_cls = get_class(f"f5_tts.model.{config['backbone']}")
    mel_spec = config["mel_spec"]
    LOGGER.info("Dispositivo: %s", selected_device)
    LOGGER.info("Arquitetura: %s %s", model_name, json.dumps(config["arch"], ensure_ascii=False))
    LOGGER.info("Tokenizer: %s", config["tokenizer"])
    LOGGER.info("Vocoder: %s", mel_spec["mel_spec_type"])

    try:
        vocoder = load_vocoder(vocoder_name=mel_spec["mel_spec_type"], device=selected_device)
    except Exception as exc:
        LOGGER.error("Falha ao carregar vocoder. Verifique internet/cache do Kaggle.")
        LOGGER.error("Traceback completo:\n%s", "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        raise

    use_ema = True
    ckpt_info = inspect_checkpoint(Path(checkpoint_path), Path(vocab_path))
    if not ckpt_info.get("has_ema"):
        use_ema = False
    LOGGER.info("Uso de EMA: %s", use_ema)

    model = load_model(
        model_cls,
        config["arch"],
        checkpoint_path,
        mel_spec_type=mel_spec["mel_spec_type"],
        vocab_file=vocab_path,
        use_ema=use_ema,
        device=selected_device,
    )

    processed_ref_audio, processed_ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text, show_info=LOGGER.info)
    final_wave, final_sample_rate, _ = infer_process(
        processed_ref_audio,
        processed_ref_text,
        text,
        model,
        vocoder,
        mel_spec_type=mel_spec["mel_spec_type"],
        cross_fade_duration=0.15,
        nfe_step=32,
        speed=1.0,
        show_info=LOGGER.info,
        device=selected_device,
    )
    if final_wave is None or len(final_wave) == 0:
        raise RuntimeError("Inferencia retornou audio vazio.")

    final_wave = np.asarray(final_wave, dtype=np.float32)
    peak = float(np.max(np.abs(final_wave)))
    rms = float(np.sqrt(np.mean(final_wave * final_wave)))
    duration = len(final_wave) / float(final_sample_rate)
    if peak <= 0 or rms <= 1e-5:
        raise RuntimeError(f"Audio gerado parece silencioso: peak={peak}, rms={rms}")

    sf.write(str(output), final_wave, final_sample_rate)
    validate_downloaded_file(output)
    LOGGER.info("Audio gerado: %s", output)
    LOGGER.info("Duracao gerada: %.3fs | peak=%.6f | rms=%.6f", duration, peak, rms)

    del model
    del vocoder
    if selected_device == "cuda":
        torch.cuda.empty_cache()
    return str(output)


def audit_voice_package(download_checkpoint: bool = False) -> tuple[VoicePackage, DiagnosticReport]:
    package = build_voice_package(download_checkpoint=download_checkpoint)
    report = DiagnosticReport()

    report.add("Manifesto", True, str(package.manifest_path))
    report.add("Checkpoint principal", package.checkpoint.remote_path in package.remote_files, package.checkpoint.remote_path)
    report.add("Arquitetura identificada", package.manifest.get("architecture") == "F5-TTS", package.manifest.get("exp_name", ""))
    report.add("Vocab encontrado", bool(package.local_vocab_path and package.local_vocab_path.exists()), str(package.local_vocab_path))

    try:
        vocab_report = validate_vocab(package.local_vocab_path, package.local_base_vocab_path)
        report.add("Vocab compativel", not vocab_report.get("duplicates"), f"{vocab_report['tokens']} tokens")
    except Exception as exc:
        report.add("Vocab compativel", False, str(exc))

    report.add("Referencia de audio", bool(package.local_reference_audio_path and package.local_reference_audio_path.exists()), str(package.local_reference_audio_path))
    ref_text = read_reference_text(package.local_reference_text_path)
    report.add("Texto da referencia", bool(ref_text), "arquivo encontrado" if ref_text else "ausente; ASR sera usado")
    report.add("Vocoder", package.config["mel_spec"]["mel_spec_type"] == "vocos", package.config["mel_spec"]["mel_spec_type"])

    try:
        import torch

        report.add("CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU disponivel")
    except Exception as exc:
        report.add("CUDA", False, str(exc))

    if download_checkpoint and package.local_checkpoint_path and package.local_checkpoint_path.exists():
        try:
            inspect_checkpoint(package.local_checkpoint_path, package.local_vocab_path)
            report.add("Checkpoint legivel", True, str(package.local_checkpoint_path))
        except Exception as exc:
            report.add("Checkpoint legivel", False, str(exc))
    else:
        report.add("Checkpoint legivel", False, "nao baixado nesta auditoria leve")

    LOGGER.info("\n%s", report.render())
    print(report.render())
    return package, report


def report_training_artifacts(package: VoicePackage) -> dict[str, Any]:
    files = package.remote_files
    inference_required = {
        "checkpoint de inferencia": package.checkpoint.remote_path,
        "vocabulario correspondente": package.vocab_remote_path,
        "configuracao da arquitetura": "recuperada de F5TTS_v1_Base no runtime f5-tts",
        "tokenizer": package.manifest.get("tokenizer"),
        "audio de referencia": package.reference_audio_remote_path,
        "transcricao da referencia": package.reference_text_remote_path,
        "vocoder e parametros de audio": "F5TTS_v1_Base/vocos/24000Hz/100mel",
    }
    training_required = {
        "checkpoint completo": package.checkpoint.remote_path,
        "estado do otimizador": None,
        "scheduler": None,
        "scaler AMP": None,
        "pesos EMA": "a confirmar pela inspecao do checkpoint",
        "contador epoch/update": "a confirmar pela inspecao do checkpoint",
        "configuracao completa do treinamento": package.setting_remote_path,
        "seed": None,
        "commit F5-TTS": None,
        "versoes das dependencias": None,
        "dataset": package.manifest.get("base_library", {}).get("repo_id"),
        "metadata/raw.arrow": None,
        "duration.json": f"{package.voice_dir}/docs/duration.json" if f"{package.voice_dir}/docs/duration.json" in files else None,
        "audios de treino": None,
        "train/validation split": None,
        "vocabulario": package.vocab_remote_path,
        "logs e metricas": f"{package.voice_dir}/docs/duration.json" if f"{package.voice_dir}/docs/duration.json" in files else None,
        "checkpoint-base exato": f"{package.base_library_dir}/{package.manifest.get('base_library', {}).get('checkpoint')}" if package.base_library_dir else None,
    }
    report = {
        "obrigatorios_inferencia": inference_required,
        "treinamento_reproducao": training_required,
        "presentes": sorted(path for path in files),
        "ausentes_inferencia": [key for key, value in inference_required.items() if not value],
        "ausentes_treinamento": [key for key, value in training_required.items() if not value],
    }
    LOGGER.info("Relatorio de artefatos: %s", json.dumps(report, ensure_ascii=False, indent=2))
    return report


def prepare_model_files(download: bool = True) -> VoicePackage:
    return build_voice_package(download_checkpoint=download)


def synthesize_for_notebook(text: str = TEST_TEXT, output_path: str = "/kaggle/working/resultado_voz.wav") -> str:
    package = build_voice_package(download_checkpoint=True)
    validate_vocab(package.local_vocab_path, package.local_base_vocab_path)
    validate_reference_audio(package.local_reference_audio_path, package.config["mel_spec"]["target_sample_rate"])
    ref_text = read_reference_text(package.local_reference_text_path)
    return generate_speech(
        text=text,
        output_path=output_path,
        checkpoint_path=str(package.local_checkpoint_path),
        vocab_path=str(package.local_vocab_path),
        ref_audio_path=str(package.local_reference_audio_path),
        ref_text=ref_text,
        model_name=package.manifest.get("exp_name", "F5TTS_v1_Base"),
        model_config=package.config,
    )


def create_gradio_app():
    import gradio as gr

    package_holder: dict[str, VoicePackage] = {}

    def generate(text: str):
        try:
            if "package" not in package_holder:
                package_holder["package"] = build_voice_package(download_checkpoint=True)
            package = package_holder["package"]
            output_path = OUTPUT_DIR / "resultado_voz_gradio.wav"
            audio_path = generate_speech(
                text=text,
                output_path=str(output_path),
                checkpoint_path=str(package.local_checkpoint_path),
                vocab_path=str(package.local_vocab_path),
                ref_audio_path=str(package.local_reference_audio_path),
                ref_text=read_reference_text(package.local_reference_text_path),
                model_name=package.manifest.get("exp_name", "F5TTS_v1_Base"),
                model_config=package.config,
            )
            return audio_path, audio_path, f"Audio gerado: {audio_path}"
        except Exception as exc:
            LOGGER.exception("Erro no Gradio")
            return None, None, f"Erro: {exc}"

    with gr.Blocks(title="Super Voz F5-TTS") as demo:
        gr.Markdown("# Super Voz F5-TTS")
        text_box = gr.Textbox(label="Texto", value=TEST_TEXT, lines=3)
        button = gr.Button("Gerar WAV")
        audio = gr.Audio(label="Audio gerado", type="filepath")
        download = gr.File(label="Download")
        status = gr.Textbox(label="Status", interactive=False)
        button.click(generate, inputs=text_box, outputs=[audio, download, status])
    return demo


def main() -> None:
    package, report = audit_voice_package(download_checkpoint=True)
    report_training_artifacts(package)
    if not report.ready:
        raise RuntimeError("Auditoria falhou; inferencia nao iniciada.")
    audio_path = synthesize_for_notebook(TEST_TEXT, "/kaggle/working/resultado_voz.wav")
    print(f"Audio gerado: {audio_path}")


if __name__ == "__main__":
    main()
