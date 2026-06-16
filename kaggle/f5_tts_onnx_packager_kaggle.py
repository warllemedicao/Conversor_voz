from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit


DEFAULT_SOURCE_URL = "https://huggingface.co/buckets/warllem/Voz_Noslen"
DEFAULT_REVISION = "main"
DEFAULT_VOICE_DIR = "voices/v_minha_voz_f5_tts_ptbr"
PACKAGER_VERSION = "2026.06.16.1"
DEFAULT_TEST_TEXT = "Boa noite Warllem, este é um teste do modo lite em CPU."


def resolve_work_root() -> Path:
    configured = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
    try:
        configured.mkdir(parents=True, exist_ok=True)
        return configured
    except OSError:
        fallback = Path(os.environ.get("TMPDIR", "/tmp")) / "voz_noslen_onnx_working"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


WORK_ROOT = resolve_work_root()
os.environ.setdefault("NUMBA_CACHE_DIR", str(WORK_ROOT / "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(WORK_ROOT / "matplotlib_cache"))
DOWNLOAD_DIR = WORK_ROOT / "voz_noslen_f5tts_snapshot"
STAGING_DIR = WORK_ROOT / "voz_noslen_onnx_package"
LOG_PATH = WORK_ROOT / "voz_noslen_onnx_packager.log"


@dataclass(frozen=True)
class PackagePaths:
    source_snapshot: Path
    staging_root: Path
    copied_training_dir: Path
    onnx_dir: Path
    scripts_dir: Path
    root_manifest_path: Path
    metadata_path: Path
    export_report_path: Path


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("voz_noslen_onnx_packager")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


LOGGER = setup_logging()


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


def repo_id_from_url_or_id(value: str) -> str:
    if not value.startswith(("http://", "https://")):
        return value.strip("/")

    parsed = urlparse(value)
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if parts[:1] in (["models"], ["model"]):
        parts = parts[1:]
    if parts[:1] in (["datasets"], ["spaces"]):
        parts = parts[1:]
    if parts[:1] == ["buckets"]:
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError(f"Nao consegui resolver repo_id a partir de: {value}")
    return "/".join(parts[:2])


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns(".git", ".cache", "__pycache__", "*.tmp")
    shutil.copytree(src, dst, ignore=ignore)


def link_or_copy_file(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def move_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def find_first(root: Path, patterns: tuple[str, ...]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.glob(pattern))
    files = sorted(path for path in matches if path.is_file())
    return files[0] if files else None


def find_manifest(root: Path) -> Path | None:
    preferred = sorted(root.glob("voices/*/manifest.json"))
    if preferred:
        return preferred[0]
    return find_first(root, ("**/manifest.json",))


def find_checkpoint(root: Path, manifest: dict[str, Any] | None, manifest_path: Path | None) -> Path:
    candidates: list[Path] = []
    if manifest and manifest_path:
        base_dir = manifest_path.parent
        for key in ("voice_checkpoint", "inference_checkpoint", "final_checkpoint", "latest_checkpoint"):
            value = manifest.get(key)
            if not value:
                continue
            candidate = root / value if str(value).startswith(("voices/", "libraries/")) else base_dir / value
            candidates.append(candidate)

    candidates.extend(
        sorted(root.glob(pattern))
        for pattern in (
            "**/model_*.pt",
            "**/latest_checkpoint.pt",
            "**/model_last.pt",
            "**/model_last.safetensors",
        )
    )
    flat_candidates: list[Path] = []
    for item in candidates:
        if isinstance(item, list):
            flat_candidates.extend(item)
        else:
            flat_candidates.append(item)

    existing = [path for path in flat_candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError("Nenhum checkpoint .pt/.safetensors encontrado no pacote F5-TTS.")
    return sorted(existing, key=lambda path: path.stat().st_size, reverse=True)[0]


def find_vocab(root: Path, checkpoint_path: Path) -> Path:
    local = checkpoint_path.parent / "vocab.txt"
    if local.is_file():
        return local
    vocab = find_first(root, ("voices/*/model/vocab.txt", "**/vocab.txt"))
    if not vocab:
        raise FileNotFoundError("Nenhum vocab.txt encontrado no pacote F5-TTS.")
    return vocab


def find_reference_audio(root: Path, voice_dir: str) -> Path | None:
    voice_ref = root / voice_dir / "data_reference" / "referencia_voz.wav"
    if voice_ref.is_file():
        return voice_ref
    voice_refs = sorted(path for path in (root / voice_dir / "data_reference").glob("*.wav") if path.is_file())
    if voice_refs:
        return voice_refs[0]
    return find_first(
        root,
        (
            "voices/*/data_reference/referencia_voz.wav",
            "voices/*/data_reference/*.wav",
            "**/referencia*.wav",
            "**/reference*.wav",
            "**/*.wav",
        ),
    )


def load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extract_reference_text(manifest: dict[str, Any] | None, manifest_path: Path | None, reference_audio_path: Path | None) -> str | None:
    if manifest:
        for key in (
            "reference_text",
            "ref_text",
            "transcript",
            "reference_transcript",
            "data_reference_text",
            "prompt_text",
        ):
            value = manifest.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("reference", "data_reference", "speaker_reference"):
            value = manifest.get(key)
            if isinstance(value, dict):
                for nested_key in ("text", "transcript", "ref_text"):
                    nested_value = value.get(nested_key)
                    if isinstance(nested_value, str) and nested_value.strip():
                        return nested_value.strip()

    candidates: list[Path] = []
    if reference_audio_path:
        candidates.extend(
            [
                reference_audio_path.with_suffix(".txt"),
                reference_audio_path.parent / "referencia_voz.txt",
                reference_audio_path.parent / "reference_text.txt",
                reference_audio_path.parent / "ref_text.txt",
                reference_audio_path.parent / "transcript.txt",
            ]
        )
    if manifest_path:
        candidates.extend(
            [
                manifest_path.parent / "reference_text.txt",
                manifest_path.parent / "ref_text.txt",
                manifest_path.parent / "transcript.txt",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text
    return None


def copy_required_runtime_files(
    paths: PackagePaths,
    checkpoint_path: Path,
    vocab_path: Path,
    reference_audio_path: Path | None,
    reference_text: str | None,
) -> dict[str, Any]:
    model_dir = paths.staging_root / "model"
    ref_dir = paths.staging_root / "reference"
    model_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    packaged_checkpoint = model_dir / checkpoint_path.name
    packaged_vocab = model_dir / "vocab.txt"
    checkpoint_storage = link_or_copy_file(checkpoint_path, packaged_checkpoint)
    vocab_storage = link_or_copy_file(vocab_path, packaged_vocab)

    packaged_reference_audio: Path | None = None
    reference_audio_storage: str | None = None
    if reference_audio_path:
        packaged_reference_audio = ref_dir / "referencia_voz.wav"
        reference_audio_storage = link_or_copy_file(reference_audio_path, packaged_reference_audio)

    reference_text_path = ref_dir / "reference_text.txt"
    if reference_text:
        reference_text_path.write_text(reference_text + "\n", encoding="utf-8")
    else:
        reference_text_path.write_text(
            "Texto de referencia nao encontrado no pacote original. O script de teste tentara transcrever a referencia automaticamente.\n",
            encoding="utf-8",
        )

    return {
        "checkpoint": packaged_checkpoint.relative_to(paths.staging_root).as_posix(),
        "vocab": packaged_vocab.relative_to(paths.staging_root).as_posix(),
        "reference_audio": packaged_reference_audio.relative_to(paths.staging_root).as_posix()
        if packaged_reference_audio
        else None,
        "reference_text": reference_text_path.relative_to(paths.staging_root).as_posix(),
        "reference_text_available": bool(reference_text),
        "storage": {
            "checkpoint": checkpoint_storage,
            "vocab": vocab_storage,
            "reference_audio": reference_audio_storage,
        },
    }


def is_bucket_url(value: str) -> bool:
    return value.startswith(("http://", "https://")) and "/buckets/" in urlparse(value).path


def strip_url_query(value: str) -> str:
    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def bucket_relative_path(file_url: str) -> Path:
    parsed = urlparse(strip_url_query(file_url))
    parts = [part for part in parsed.path.split("/") if part]
    for marker in ("resolve", "raw", "blob"):
        if marker in parts:
            index = parts.index(marker)
            if len(parts) > index + 1:
                remaining = parts[index + 1 :]
                if remaining and remaining[0] in ("main", "refs", "resolve"):
                    remaining = remaining[1:]
                return Path(*remaining)
    if "buckets" in parts and len(parts) > parts.index("buckets") + 3:
        index = parts.index("buckets")
        return Path(*parts[index + 3 :])
    return Path(parts[-1])


def is_tmp_or_partial(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".tmp", ".partial", ".incomplete"))


def choose_bucket_checkpoint(relative_paths: list[Path], voice_dir: str) -> Path | None:
    voice_prefix = Path(voice_dir)
    preferred_names = (
        "model/model_2000.pt",
        "model/latest_checkpoint.pt",
        "model/model_last.pt",
        "model/model_last.safetensors",
        "model/final_checkpoint.pt",
    )
    available = {path.as_posix(): path for path in relative_paths}
    for name in preferred_names:
        candidate = (voice_prefix / name).as_posix()
        if candidate in available:
            return available[candidate]

    checkpoints = [
        path
        for path in relative_paths
        if path.as_posix().startswith(f"{voice_dir.rstrip('/')}/model/")
        and path.suffix.lower() in (".pt", ".safetensors")
        and "base_checkpoint" not in path.name
        and not is_tmp_or_partial(path)
    ]
    return sorted(checkpoints)[0] if checkpoints else None


def filter_bucket_files(file_urls: set[str], voice_dir: str, download_mode: str) -> list[tuple[str, Path]]:
    entries = [(url, bucket_relative_path(url)) for url in file_urls]
    entries = [(url, path) for url, path in entries if not is_tmp_or_partial(path)]
    if download_mode == "all":
        return sorted(entries, key=lambda item: item[1].as_posix())

    relative_paths = [path for _, path in entries]
    checkpoint = choose_bucket_checkpoint(relative_paths, voice_dir)
    if checkpoint is None:
        raise RuntimeError(f"Nenhum checkpoint principal encontrado em {voice_dir}/model.")

    wanted: set[str] = {checkpoint.as_posix()}
    voice_prefix = voice_dir.rstrip("/")
    for _, path in entries:
        path_text = path.as_posix()
        if path_text == ".gitattributes":
            wanted.add(path_text)
        if path_text.startswith(f"{voice_prefix}/"):
            if path_text.endswith((".md", ".json", ".txt", ".wav")):
                wanted.add(path_text)
            if path_text == f"{voice_prefix}/model/vocab.txt":
                wanted.add(path_text)
        if path_text.startswith("libraries/"):
            if path_text.endswith((".md", ".json", ".txt", ".wav")):
                wanted.add(path_text)

    LOGGER.info("Modo essential: checkpoint escolhido para download: %s", checkpoint.as_posix())
    LOGGER.info("Modo essential: %s arquivo(s) selecionado(s), checkpoints duplicados e .tmp ignorados.", len(wanted))
    return sorted((url, path) for url, path in entries if path.as_posix() in wanted)


def download_http_file(url: str, output_path: Path, token: str | None) -> None:
    import requests

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def download_bucket_source(source_url: str, token: str | None, voice_dir: str, download_mode: str) -> Path:
    import re
    import requests

    clean_dir(DOWNLOAD_DIR)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    source_url = source_url.rstrip("/")
    pages = [source_url, f"{source_url}/tree/main"]
    seen_pages: set[str] = set()
    file_urls: set[str] = set()

    LOGGER.info("Origem parece ser Hugging Face Buckets; tentando listar links HTML em %s", source_url)
    while pages:
        page_url = pages.pop(0)
        if page_url in seen_pages:
            continue
        seen_pages.add(page_url)
        response = requests.get(page_url, headers=headers, timeout=60)
        if response.status_code == 404:
            continue
        response.raise_for_status()

        for href in re.findall(r'href=["\']([^"\']+)["\']', response.text):
            absolute = strip_url_query(urljoin(page_url, href))
            parsed = urlparse(absolute)
            if parsed.netloc != urlparse(source_url).netloc:
                continue
            if "/tree/" in parsed.path and "/buckets/" in parsed.path and absolute not in seen_pages:
                pages.append(absolute)
            if any(marker in parsed.path for marker in ("/resolve/", "/raw/", "/blob/")) and "/buckets/" in parsed.path:
                file_urls.add(absolute.replace("/blob/", "/resolve/").replace("/raw/", "/resolve/"))

    if not file_urls:
        raise RuntimeError(
            "Nao consegui listar arquivos do link /buckets/. Esse endereco nao e um Model Repo padrao do Hugging Face. "
            "Abra o bucket no navegador, copie os arquivos para um Model Repo normal ou informe o repo_id correto em --repo-id. "
            f"Origem recebida: {source_url}"
        )

    selected_files = filter_bucket_files(file_urls, voice_dir, download_mode)
    for url, relative in selected_files:
        output_path = DOWNLOAD_DIR / relative
        LOGGER.info("Baixando bucket: %s -> %s", url, output_path)
        download_http_file(url, output_path, token)

    return DOWNLOAD_DIR


def download_source_repo(repo_id: str, revision: str, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    clean_dir(DOWNLOAD_DIR)
    LOGGER.info("Baixando snapshot de %s @ %s para %s", repo_id, revision, DOWNLOAD_DIR)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            local_dir=str(DOWNLOAD_DIR),
            token=token,
            ignore_patterns=(".git/*",),
        )
    except TypeError:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            local_dir=str(DOWNLOAD_DIR),
            token=token,
        )
    return DOWNLOAD_DIR


def download_source(
    source: str,
    repo_id: str | None,
    revision: str,
    token: str | None,
    voice_dir: str,
    download_mode: str,
) -> tuple[Path, str]:
    if repo_id:
        return download_source_repo(repo_id, revision, token), repo_id
    if is_bucket_url(source):
        return download_bucket_source(source, token, voice_dir, download_mode), source
    resolved_repo_id = repo_id_from_url_or_id(source)
    return download_source_repo(resolved_repo_id, revision, token), resolved_repo_id


def make_package_paths() -> PackagePaths:
    clean_dir(STAGING_DIR)
    copied_training_dir = STAGING_DIR / "f5_tts_original"
    onnx_dir = STAGING_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = STAGING_DIR / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    return PackagePaths(
        source_snapshot=DOWNLOAD_DIR,
        staging_root=STAGING_DIR,
        copied_training_dir=copied_training_dir,
        onnx_dir=onnx_dir,
        scripts_dir=scripts_dir,
        root_manifest_path=STAGING_DIR / "manifest.json",
        metadata_path=STAGING_DIR / "package_metadata.json",
        export_report_path=STAGING_DIR / "onnx_export_report.json",
    )


def build_default_f5_config(manifest: dict[str, Any] | None) -> dict[str, Any]:
    exp_name = (manifest or {}).get("exp_name") or "F5TTS_v1_Base"
    if exp_name != "F5TTS_v1_Base":
        raise RuntimeError(f"Exportador preparado apenas para F5TTS_v1_Base; encontrado: {exp_name!r}")
    return {
        "exp_name": exp_name,
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
        "tokenizer": (manifest or {}).get("tokenizer") or "char",
    }


def infer_module_float_dtype(module: Any) -> Any:
    import torch

    for parameter in module.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    for buffer in module.buffers():
        if buffer.is_floating_point():
            return buffer.dtype
    return torch.float32


class F5TransformerOnnxWrapper:
    def __init__(self, model: Any, compute_dtype: Any) -> None:
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self, inner_model: Any, dtype: Any) -> None:
                super().__init__()
                self.inner_model = inner_model
                self.transformer = getattr(inner_model, "transformer", inner_model)
                self.compute_dtype = dtype

            def _cast_float_input(self, value):
                if torch.is_tensor(value) and value.is_floating_point() and value.dtype != self.compute_dtype:
                    return value.to(dtype=self.compute_dtype)
                return value

            def forward(self, x, cond, text, time, mask):
                x = self._cast_float_input(x)
                cond = self._cast_float_input(cond)
                time = self._cast_float_input(time)
                try:
                    return self.transformer(
                        x=x,
                        cond=cond,
                        text=text,
                        time=time,
                        mask=mask,
                        drop_audio_cond=False,
                        drop_text=False,
                    )
                except TypeError:
                    return self.transformer(
                        x,
                        cond,
                        text,
                        time,
                        mask=mask,
                        drop_audio_cond=False,
                        drop_text=False,
                    )

        self.module = Wrapper(model, compute_dtype)


def kwargs_supported_by(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def legacy_onnx_export(
    model: Any,
    sample_inputs: tuple[Any, ...],
    onnx_path: Path,
    input_names: list[str],
    output_names: list[str],
) -> str:
    import torch

    common_kwargs = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": 18,
        "do_constant_folding": True,
    }

    try:
        from torch.onnx import utils as onnx_utils

        LOGGER.info("Tentando exportacao ONNX pelo caminho legado torch.onnx.utils.export.")
        legacy_kwargs = {
            **common_kwargs,
            "use_external_data_format": True,
        }
        onnx_utils.export(
            model,
            sample_inputs,
            str(onnx_path),
            **kwargs_supported_by(onnx_utils.export, legacy_kwargs),
        )
        return "torch.onnx.utils.export"
    except Exception as exc:
        LOGGER.warning("Falha no caminho legado direto: %s: %s", type(exc).__name__, exc)

    try:
        LOGGER.info("Tentando exportacao ONNX por torch.onnx.export com dynamo=False.")
        export_kwargs = {
            **common_kwargs,
            "dynamo": False,
            "external_data": True,
        }
        torch.onnx.export(
            model,
            sample_inputs,
            str(onnx_path),
            **kwargs_supported_by(torch.onnx.export, export_kwargs),
        )
        return "torch.onnx.export_dynamo_false"
    except Exception as exc:
        LOGGER.warning("Falha em torch.onnx.export com dynamo=False: %s: %s", type(exc).__name__, exc)

    LOGGER.info("Tentando torch.jit.trace antes da exportacao ONNX.")
    traced = torch.jit.trace(model, sample_inputs, strict=False)
    traced.eval()
    export_kwargs = {
        **common_kwargs,
        "dynamo": False,
        "external_data": True,
    }
    torch.onnx.export(
        traced,
        sample_inputs,
        str(onnx_path),
        **kwargs_supported_by(torch.onnx.export, export_kwargs),
    )
    return "torch.jit.trace_then_torch.onnx.export"


def export_f5_core_to_onnx(checkpoint_path: Path, vocab_path: Path, onnx_dir: Path, manifest: dict[str, Any] | None) -> dict[str, Any]:
    import torch
    from f5_tts.infer.utils_infer import load_model
    from hydra.utils import get_class

    try:
        import onnxscript  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Dependencia ausente: onnxscript. Rode novamente a celula 2 do notebook atualizado "
            "ou execute: pip install onnxscript"
        ) from exc

    config = build_default_f5_config(manifest)
    model_cls = get_class(f"f5_tts.model.{config['backbone']}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    onnx_path = onnx_dir / "f5_tts_transformer_core.onnx"

    LOGGER.info("Carregando F5-TTS em %s para exportacao ONNX", device)
    try:
        model = load_model(
            model_cls,
            config["arch"],
            str(checkpoint_path),
            mel_spec_type=config["mel_spec"]["mel_spec_type"],
            vocab_file=str(vocab_path),
            use_ema=True,
            device=device,
        )
        use_ema = True
    except Exception:
        LOGGER.warning("Falha ao carregar com EMA; tentando carregar pesos sem EMA.")
        model = load_model(
            model_cls,
            config["arch"],
            str(checkpoint_path),
            mel_spec_type=config["mel_spec"]["mel_spec_type"],
            vocab_file=str(vocab_path),
            use_ema=False,
            device=device,
        )
        use_ema = False
    model.eval()
    model_compute_dtype = infer_module_float_dtype(model)
    LOGGER.info("Dtype de ponto flutuante do modelo: %s", model_compute_dtype)
    wrapped = F5TransformerOnnxWrapper(model, model_compute_dtype).module.to(device).eval()

    batch = 1
    frames = 64
    text_tokens = 32
    mel_channels = config["mel_spec"]["n_mel_channels"]
    x = torch.randn(batch, frames, mel_channels, device=device)
    cond = torch.zeros(batch, frames, mel_channels, device=device)
    text = torch.randint(0, max(2, sum(1 for _ in vocab_path.open(encoding="utf-8"))), (batch, text_tokens), device=device)
    time = torch.full((batch,), 0.5, dtype=torch.float32, device=device)
    mask = torch.ones(batch, frames, dtype=torch.bool, device=device)

    LOGGER.info("Exportando nucleo Transformer para %s", onnx_path)
    export_method = legacy_onnx_export(
        wrapped,
        (x, cond, text, time, mask),
        onnx_path,
        ["x", "cond", "text", "time", "mask"],
        ["pred"],
    )

    report: dict[str, Any] = {
        "status": "ok",
        "packager_version": PACKAGER_VERSION,
        "onnx_file": str(onnx_path),
        "checkpoint": str(checkpoint_path),
        "vocab": str(vocab_path),
        "device": device,
        "use_ema": use_ema,
        "model_compute_dtype": str(model_compute_dtype),
        "export_mode": "legacy_static",
        "export_method": export_method,
        "static_shapes": {"batch": batch, "frames": frames, "text_tokens": text_tokens, "mel_channels": mel_channels},
        "note": "Este ONNX contem o nucleo Transformer/DiT do F5-TTS com formas estaticas. O pacote mantem os arquivos originais para inferencia Python completa.",
    }
    validate_onnx(onnx_path, report)
    return report


def validate_onnx(onnx_path: Path, report: dict[str, Any]) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    report["onnx_checker"] = "ok"
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        report["onnxruntime_inputs"] = [
            {"name": item.name, "shape": item.shape, "type": item.type} for item in session.get_inputs()
        ]
        report["onnxruntime_outputs"] = [
            {"name": item.name, "shape": item.shape, "type": item.type} for item in session.get_outputs()
        ]
        report["onnxruntime_load"] = "ok"
    except Exception as exc:
        report["onnxruntime_load"] = f"falhou: {type(exc).__name__}: {exc}"


def write_cpu_test_script(paths: PackagePaths) -> Path:
    script_path = paths.scripts_dir / "test_package_cpu.py"
    script = r'''
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf


DEFAULT_TEXT = "Boa noite Warllem, este é um teste do modo lite em CPU."

PACKAGE_DEFAULT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(PACKAGE_DEFAULT_ROOT / ".cache" / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_DEFAULT_ROOT / ".cache" / "matplotlib"))


def ort_dtype(type_name: str):
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(bool)": np.bool_,
    }
    return mapping.get(type_name, np.float32)


def concrete_shape(shape):
    return [1 if not isinstance(dim, int) or dim <= 0 else dim for dim in shape]


def run_onnx_smoke(package_dir: Path, report: dict) -> dict:
    import onnxruntime as ort

    onnx_files = sorted((package_dir / "onnx").glob("*.onnx"))
    results = []
    for onnx_file in onnx_files:
        session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        feeds = {}
        input_details = []
        for item in session.get_inputs():
            dtype = ort_dtype(item.type)
            shape = concrete_shape(item.shape)
            if dtype == np.bool_:
                value = np.ones(shape, dtype=dtype)
            elif np.issubdtype(dtype, np.integer):
                value = np.ones(shape, dtype=dtype)
            else:
                value = np.zeros(shape, dtype=dtype)
            feeds[item.name] = value
            input_details.append({"name": item.name, "shape": item.shape, "type": item.type})
        start = time.perf_counter()
        outputs = session.run(None, feeds)
        elapsed = time.perf_counter() - start
        results.append(
            {
                "file": onnx_file.relative_to(package_dir).as_posix(),
                "elapsed_seconds": elapsed,
                "inputs": input_details,
                "outputs": [
                    {
                        "name": item.name,
                        "shape": item.shape,
                        "type": item.type,
                        "actual_shape": list(outputs[index].shape),
                        "actual_dtype": str(outputs[index].dtype),
                    }
                    for index, item in enumerate(session.get_outputs())
                ],
            }
        )
    report["onnxruntime_cpu_smoke_test"] = {"status": "ok", "models": results}
    return report


def read_reference_text(path: Path) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("Texto de referencia nao encontrado"):
        return ""
    return text


def run_f5_cpu_inference(package_dir: Path, text: str, output_wav: Path, nfe_step: int, speed: float, report: dict) -> dict:
    from importlib.resources import files

    from f5_tts.infer.utils_infer import (
        infer_process,
        load_model,
        load_vocoder,
        preprocess_ref_audio_text,
    )
    from hydra.utils import get_class
    from omegaconf import OmegaConf

    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = package_dir / manifest["runtime_files"]["checkpoint"]
    vocab = package_dir / manifest["runtime_files"]["vocab"]
    ref_audio = package_dir / manifest["runtime_files"]["reference_audio"]
    ref_text = read_reference_text(package_dir / manifest["runtime_files"]["reference_text"])
    model_name = manifest.get("f5_tts_model", "F5TTS_v1_Base")
    vocoder_name = manifest.get("vocoder", "vocos")

    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    start = time.perf_counter()
    vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False, device="cpu")
    ema_model = load_model(
        model_cls,
        model_arc,
        str(checkpoint),
        mel_spec_type=vocoder_name,
        vocab_file=str(vocab),
        device="cpu",
    )
    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(str(ref_audio), ref_text)
    audio, sample_rate, _ = infer_process(
        ref_audio_processed,
        ref_text_processed,
        text,
        ema_model,
        vocoder,
        mel_spec_type=vocoder_name,
        nfe_step=nfe_step,
        speed=speed,
        device="cpu",
    )
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), audio, sample_rate)
    elapsed = time.perf_counter() - start

    info = sf.info(str(output_wav))
    report["wav_generation_cpu_test"] = {
        "status": "ok",
        "text": text,
        "output_wav": output_wav.relative_to(package_dir).as_posix(),
        "elapsed_seconds": elapsed,
        "sample_rate": sample_rate,
        "frames": info.frames,
        "duration_seconds": info.duration,
        "nfe_step": nfe_step,
        "speed": speed,
        "runtime": "F5-TTS Python + vocos on CPU; ONNX is used only for DiT/core smoke validation.",
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Testa o pacote Voz_Noslen F5-TTS ONNX em CPU.")
    parser.add_argument("--package-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output-wav", default="test_outputs/voz_noslen_lite_cpu.wav")
    parser.add_argument("--nfe-step", type=int, default=4)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--report", default="onnx_export_report.json")
    args = parser.parse_args()

    package_dir = Path(args.package_dir).resolve()
    report_path = package_dir / args.report
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    report = run_onnx_smoke(package_dir, report)
    output_wav = Path(args.output_wav)
    if not output_wav.is_absolute():
        output_wav = package_dir / output_wav
    report = run_f5_cpu_inference(package_dir, args.text, output_wav, args.nfe_step, args.speed, report)
    report["cpu_test_command"] = (
        "python scripts/test_package_cpu.py "
        f"--text {args.text!r} --output-wav {str(Path(args.output_wav))!r} "
        f"--nfe-step {args.nfe_step} --speed {args.speed}"
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["wav_generation_cpu_test"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''
    script_path.write_text(textwrap.dedent(script).lstrip(), encoding="utf-8")
    return script_path


def write_root_manifest(
    paths: PackagePaths,
    source_label: str,
    revision: str,
    hf_folder: str,
    runtime_files: dict[str, Any],
    export_report: dict[str, Any],
) -> None:
    manifest = {
        "name": "Voz_Noslen F5-TTS ONNX/Lite package",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "packager_version": PACKAGER_VERSION,
        "source": source_label,
        "source_revision": revision,
        "target_huggingface_folder": hf_folder,
        "f5_tts_model": "F5TTS_v1_Base",
        "sample_rate": 24000,
        "vocoder": "vocos",
        "runtime_files": runtime_files,
        "onnx_files": sorted(path.relative_to(paths.staging_root).as_posix() for path in paths.onnx_dir.glob("*.onnx")),
        "test_script": "scripts/test_package_cpu.py",
        "lite_contract_status": "partial_pipeline",
        "lite_contract": {
            "available_onnx": "DiT/Transformer core only: inputs x, cond, text, time, mask; output pred.",
            "full_text_to_audio_onnx": False,
            "required_runtime": "Python preprocessing/sampling/postprocessing from f5-tts plus vocos runtime.",
        },
        "limitations": [
            "F5-TTS inference includes tokenizer/preprocess, reference-audio conditioning, iterative flow-matching sampling, vocoder and WAV writing.",
            "The exported ONNX is not a single text-to-waveform graph and cannot satisfy a high-level text/text_ids -> audio backend contract by itself.",
            "The CPU WAV test uses f5-tts Python + vocos for the full pipeline and onnxruntime only to validate the exported DiT/core graph.",
            "Reference text is used when present; otherwise F5-TTS preprocessing may invoke automatic transcription for the reference audio.",
        ],
        "onnx_export_summary": export_report,
    }
    paths.root_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def list_package_files(paths: PackagePaths) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in paths.staging_root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(paths.staging_root).as_posix(),
                "size_bytes": path.stat().st_size,
            }
        )
    return files


def run_cpu_package_test(paths: PackagePaths, test_text: str, nfe_step_value: int, speed_value: float) -> dict[str, Any]:
    command = [
        sys.executable,
        str(paths.scripts_dir / "test_package_cpu.py"),
        "--package-dir",
        str(paths.staging_root),
        "--text",
        test_text,
        "--output-wav",
        "test_outputs/voz_noslen_lite_cpu.wav",
        "--nfe-step",
        str(nfe_step_value),
        "--speed",
        str(speed_value),
    ]
    LOGGER.info("Rodando teste CPU do pacote: %s", " ".join(command))
    start = datetime.now(timezone.utc)
    completed = subprocess.run(command, cwd=str(paths.staging_root), text=True, capture_output=True, check=False)
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    result = {
        "command": " ".join(command),
        "elapsed_seconds": elapsed,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(f"Teste CPU falhou com codigo {completed.returncode}. stderr: {completed.stderr[-2000:]}")
    return result


def write_package_metadata(
    paths: PackagePaths,
    repo_id: str,
    revision: str,
    hf_folder: str,
    checkpoint_path: Path,
    vocab_path: Path,
    reference_audio_path: Path | None,
    manifest_path: Path | None,
    manifest: dict[str, Any] | None,
    export_report: dict[str, Any],
) -> None:
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo_id": repo_id,
        "source_revision": revision,
        "target_huggingface_folder": hf_folder,
        "policy": "Arquivos originais copiados; nenhum arquivo do treinamento remoto e alterado.",
        "copied_training_dir": str(paths.copied_training_dir),
        "checkpoint": str(checkpoint_path.relative_to(paths.copied_training_dir)),
        "vocab": str(vocab_path.relative_to(paths.copied_training_dir)),
        "reference_audio": str(reference_audio_path.relative_to(paths.copied_training_dir)) if reference_audio_path else None,
        "manifest": str(manifest_path.relative_to(paths.copied_training_dir)) if manifest_path else None,
        "manifest_summary": manifest or {},
        "onnx_export": export_report,
    }
    paths.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def upload_package(paths: PackagePaths, repo_id: str, revision: str, hf_folder: str, token: str | None) -> None:
    from huggingface_hub import HfApi

    if not token:
        raise RuntimeError("HF_TOKEN ausente. Crie um Kaggle Secret chamado HF_TOKEN para enviar ao Hugging Face.")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    LOGGER.info("Enviando pacote para %s/%s", repo_id, hf_folder)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        folder_path=str(paths.staging_root),
        path_in_repo=hf_folder.strip("/"),
        commit_message=f"Add F5-TTS ONNX package at {hf_folder}",
    )


def package_voice(args: argparse.Namespace) -> None:
    LOGGER.info("Voz_Noslen ONNX packager versao: %s", PACKAGER_VERSION)
    token = get_hf_token()
    upload_repo_id = args.upload_repo_id or args.repo_id
    revision = args.revision
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    hf_folder = args.hf_folder or f"onnx_packages/voz_noslen_f5tts_onnx_{timestamp}"

    source, source_label = download_source(args.source, args.repo_id, revision, token, args.voice_dir, args.download_mode)
    paths = make_package_paths()
    move_tree(source, paths.copied_training_dir)

    manifest_path = find_manifest(paths.copied_training_dir)
    manifest = load_json_if_exists(manifest_path)
    checkpoint_path = find_checkpoint(paths.copied_training_dir, manifest, manifest_path)
    vocab_path = find_vocab(paths.copied_training_dir, checkpoint_path)
    reference_audio_path = find_reference_audio(paths.copied_training_dir, args.voice_dir)
    reference_text = extract_reference_text(manifest, manifest_path, reference_audio_path)

    LOGGER.info("Checkpoint escolhido: %s", checkpoint_path)
    LOGGER.info("Vocab escolhido: %s", vocab_path)
    LOGGER.info("Referencia de audio: %s", reference_audio_path or "nao encontrada")
    LOGGER.info("Texto de referencia: %s", "encontrado" if reference_text else "nao encontrado")

    if not reference_audio_path:
        raise FileNotFoundError("Audio de referencia obrigatorio nao encontrado; pacote Lite nao pode ser testado.")

    runtime_files = copy_required_runtime_files(paths, checkpoint_path, vocab_path, reference_audio_path, reference_text)
    test_script_path = write_cpu_test_script(paths)

    export_report: dict[str, Any]
    try:
        export_report = export_f5_core_to_onnx(checkpoint_path, vocab_path, paths.onnx_dir, manifest)
    except Exception as exc:
        export_report = {
            "status": "failed",
            "packager_version": PACKAGER_VERSION,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "note": "A copia dos arquivos originais foi preservada. Corrija a exportacao antes de usar este pacote como ONNX.",
        }
        paths.export_report_path.write_text(json.dumps(export_report, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.allow_failed_export:
            LOGGER.error("Exportacao ONNX falhou. Relatorio: %s", paths.export_report_path)
            raise

    export_report["pipeline_contract"] = {
        "full_text_to_audio_onnx_available": False,
        "reason": (
            "O F5-TTS completo depende de preprocessamento/tokenizacao, condicionamento por audio de referencia, "
            "loop iterativo de flow matching/sampling e vocoder. O arquivo ONNX exportado cobre apenas o nucleo DiT."
        ),
        "backend_lite_compatibility": "Nao compativel com um backend que espera um unico ONNX text/text_ids -> waveform.",
        "documented_pipeline": [
            "1. preprocess/tokenizer em Python via f5-tts",
            "2. DiT/Transformer core ONNX para validacao isolada do nucleo",
            "3. sampling F5-TTS em Python",
            "4. vocoder vocos em runtime Python",
            "5. escrita WAV por soundfile",
        ],
    }
    export_report["test_script"] = test_script_path.relative_to(paths.staging_root).as_posix()
    export_report["required_runtime_files"] = runtime_files
    paths.export_report_path.write_text(json.dumps(export_report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_root_manifest(paths, source_label, revision, hf_folder, runtime_files, export_report)

    cpu_test_result: dict[str, Any] | None = None
    if args.run_cpu_test:
        try:
            cpu_test_result = run_cpu_package_test(paths, args.test_text, args.test_nfe_step, args.test_speed)
            export_report = load_json_if_exists(paths.export_report_path) or export_report
            export_report["packager_cpu_test"] = cpu_test_result
        except Exception as exc:
            export_report["packager_cpu_test"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "note": "Pacote nao deve ser publicado como validado ate este teste passar em CPU.",
            }
            paths.export_report_path.write_text(json.dumps(export_report, ensure_ascii=False, indent=2), encoding="utf-8")
            if not args.allow_failed_cpu_test:
                raise
    else:
        export_report["packager_cpu_test"] = {
            "status": "skipped",
            "note": "Use scripts/test_package_cpu.py para validar onnxruntime CPU e gerar WAV antes de publicar.",
        }

    export_report["generated_files"] = list_package_files(paths)
    paths.export_report_path.write_text(json.dumps(export_report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_root_manifest(paths, source_label, revision, hf_folder, runtime_files, export_report)
    write_package_metadata(
        paths,
        source_label,
        revision,
        hf_folder,
        checkpoint_path,
        vocab_path,
        reference_audio_path,
        manifest_path,
        manifest,
        export_report,
    )

    if args.upload:
        if not upload_repo_id:
            raise RuntimeError(
                "Para fazer upload, informe --upload-repo-id com o Model Repo de destino. "
                "Buckets nao aceitam upload via HfApi.upload_folder."
            )
        upload_package(paths, upload_repo_id, revision, hf_folder, token)
    else:
        LOGGER.info("Upload desativado. Pacote local pronto em %s", paths.staging_root)

    LOGGER.info("Pacote final: %s", paths.staging_root)
    LOGGER.info("Pasta alvo no Hugging Face: %s", hf_folder)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Empacota uma voz F5-TTS em um pacote ONNX no Kaggle.")
    parser.add_argument("--source", default=os.environ.get("HF_SOURCE_URL", DEFAULT_SOURCE_URL), help="URL ou repo_id do Hugging Face.")
    parser.add_argument("--repo-id", default=os.environ.get("HF_SOURCE_REPO_ID"), help="Repo ID explicito, ex: warllem/Voz_Noslen.")
    parser.add_argument("--upload-repo-id", default=os.environ.get("HF_UPLOAD_REPO_ID"), help="Model Repo onde a pasta nova sera enviada.")
    parser.add_argument("--voice-dir", default=os.environ.get("HF_VOICE_DIR", DEFAULT_VOICE_DIR), help="Pasta da voz dentro do bucket/repo.")
    parser.add_argument(
        "--download-mode",
        choices=("essential", "all"),
        default=os.environ.get("HF_DOWNLOAD_MODE", "essential"),
        help="essential baixa apenas a voz escolhida e evita checkpoints duplicados; all tenta baixar tudo.",
    )
    parser.add_argument("--revision", default=os.environ.get("HF_REVISION", DEFAULT_REVISION))
    parser.add_argument("--hf-folder", default=os.environ.get("HF_TARGET_FOLDER"), help="Nova pasta dentro do repo Hugging Face.")
    parser.add_argument("--upload", action="store_true", default=os.environ.get("HF_UPLOAD", "1") == "1")
    parser.add_argument("--no-upload", action="store_false", dest="upload")
    parser.add_argument(
        "--allow-failed-export",
        action="store_true",
        help="Ainda cria e envia o pacote copiado se a exportacao ONNX falhar. Nao recomendado para pacote final.",
    )
    parser.add_argument("--test-text", default=os.environ.get("F5_ONNX_TEST_TEXT", DEFAULT_TEST_TEXT))
    parser.add_argument("--test-nfe-step", type=int, default=int(os.environ.get("F5_ONNX_TEST_NFE_STEP", "4")))
    parser.add_argument("--test-speed", type=float, default=float(os.environ.get("F5_ONNX_TEST_SPEED", "1.0")))
    parser.add_argument("--run-cpu-test", action="store_true", default=os.environ.get("F5_ONNX_RUN_CPU_TEST", "1") == "1")
    parser.add_argument("--skip-cpu-test", action="store_false", dest="run_cpu_test")
    parser.add_argument(
        "--allow-failed-cpu-test",
        action="store_true",
        help="Permite criar o pacote mesmo se o teste CPU falhar. Nao use para publicacao final validada.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    package_voice(args)


if __name__ == "__main__":
    main()
