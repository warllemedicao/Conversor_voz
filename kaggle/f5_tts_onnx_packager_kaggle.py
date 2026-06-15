from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit


DEFAULT_SOURCE_URL = "https://huggingface.co/buckets/warllem/Voz_Noslen"
DEFAULT_REVISION = "main"
WORK_ROOT = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
DOWNLOAD_DIR = WORK_ROOT / "voz_noslen_f5tts_snapshot"
STAGING_DIR = WORK_ROOT / "voz_noslen_onnx_package"
LOG_PATH = WORK_ROOT / "voz_noslen_onnx_packager.log"


@dataclass(frozen=True)
class PackagePaths:
    source_snapshot: Path
    staging_root: Path
    copied_training_dir: Path
    onnx_dir: Path
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


def find_reference_audio(root: Path) -> Path | None:
    return find_first(
        root,
        (
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
            if len(parts) > index + 2:
                return Path(*parts[index + 2 :])
    if "buckets" in parts and len(parts) > parts.index("buckets") + 3:
        index = parts.index("buckets")
        return Path(*parts[index + 3 :])
    return Path(parts[-1])


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


def download_bucket_source(source_url: str, token: str | None) -> Path:
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

    for url in sorted(file_urls):
        relative = bucket_relative_path(url)
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


def download_source(source: str, repo_id: str | None, revision: str, token: str | None) -> tuple[Path, str]:
    if repo_id:
        return download_source_repo(repo_id, revision, token), repo_id
    if is_bucket_url(source):
        return download_bucket_source(source, token), source
    resolved_repo_id = repo_id_from_url_or_id(source)
    return download_source_repo(resolved_repo_id, revision, token), resolved_repo_id


def make_package_paths() -> PackagePaths:
    clean_dir(STAGING_DIR)
    copied_training_dir = STAGING_DIR / "f5_tts_original"
    onnx_dir = STAGING_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    return PackagePaths(
        source_snapshot=DOWNLOAD_DIR,
        staging_root=STAGING_DIR,
        copied_training_dir=copied_training_dir,
        onnx_dir=onnx_dir,
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


class F5TransformerOnnxWrapper:
    def __init__(self, model: Any) -> None:
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self, inner_model: Any) -> None:
                super().__init__()
                self.inner_model = inner_model
                self.transformer = getattr(inner_model, "transformer", inner_model)

            def forward(self, x, cond, text, time, mask):
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

        self.module = Wrapper(model)


def export_f5_core_to_onnx(checkpoint_path: Path, vocab_path: Path, onnx_dir: Path, manifest: dict[str, Any] | None) -> dict[str, Any]:
    import torch
    from f5_tts.infer.utils_infer import load_model
    from hydra.utils import get_class

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
    wrapped = F5TransformerOnnxWrapper(model).module.to(device).eval()

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
    torch.onnx.export(
        wrapped,
        (x, cond, text, time, mask),
        str(onnx_path),
        input_names=["x", "cond", "text", "time", "mask"],
        output_names=["pred"],
        dynamic_axes={
            "x": {0: "batch", 1: "frames"},
            "cond": {0: "batch", 1: "frames"},
            "text": {0: "batch", 1: "text_tokens"},
            "time": {0: "batch"},
            "mask": {0: "batch", 1: "frames"},
            "pred": {0: "batch", 1: "frames"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    report: dict[str, Any] = {
        "status": "ok",
        "onnx_file": str(onnx_path),
        "checkpoint": str(checkpoint_path),
        "vocab": str(vocab_path),
        "device": device,
        "use_ema": use_ema,
        "note": "Este ONNX contem o nucleo Transformer/DiT do F5-TTS. O pacote mantem os arquivos originais para inferencia Python completa.",
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
        report["onnxruntime_inputs"] = [item.name for item in session.get_inputs()]
        report["onnxruntime_outputs"] = [item.name for item in session.get_outputs()]
        report["onnxruntime_load"] = "ok"
    except Exception as exc:
        report["onnxruntime_load"] = f"falhou: {type(exc).__name__}: {exc}"


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
    token = get_hf_token()
    upload_repo_id = args.upload_repo_id or args.repo_id
    revision = args.revision
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    hf_folder = args.hf_folder or f"onnx_packages/voz_noslen_f5tts_onnx_{timestamp}"

    source, source_label = download_source(args.source, args.repo_id, revision, token)
    paths = make_package_paths()
    copy_tree(source, paths.copied_training_dir)

    manifest_path = find_manifest(paths.copied_training_dir)
    manifest = load_json_if_exists(manifest_path)
    checkpoint_path = find_checkpoint(paths.copied_training_dir, manifest, manifest_path)
    vocab_path = find_vocab(paths.copied_training_dir, checkpoint_path)
    reference_audio_path = find_reference_audio(paths.copied_training_dir)

    LOGGER.info("Checkpoint escolhido: %s", checkpoint_path)
    LOGGER.info("Vocab escolhido: %s", vocab_path)
    LOGGER.info("Referencia de audio: %s", reference_audio_path or "nao encontrada")

    export_report: dict[str, Any]
    try:
        export_report = export_f5_core_to_onnx(checkpoint_path, vocab_path, paths.onnx_dir, manifest)
    except Exception as exc:
        export_report = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "note": "A copia dos arquivos originais foi preservada. Corrija a exportacao antes de usar este pacote como ONNX.",
        }
        paths.export_report_path.write_text(json.dumps(export_report, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.allow_failed_export:
            LOGGER.error("Exportacao ONNX falhou. Relatorio: %s", paths.export_report_path)
            raise

    paths.export_report_path.write_text(json.dumps(export_report, ensure_ascii=False, indent=2), encoding="utf-8")
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
    parser.add_argument("--revision", default=os.environ.get("HF_REVISION", DEFAULT_REVISION))
    parser.add_argument("--hf-folder", default=os.environ.get("HF_TARGET_FOLDER"), help="Nova pasta dentro do repo Hugging Face.")
    parser.add_argument("--upload", action="store_true", default=os.environ.get("HF_UPLOAD", "1") == "1")
    parser.add_argument("--no-upload", action="store_false", dest="upload")
    parser.add_argument(
        "--allow-failed-export",
        action="store_true",
        help="Ainda cria e envia o pacote copiado se a exportacao ONNX falhar. Nao recomendado para pacote final.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    package_voice(args)


if __name__ == "__main__":
    main()
