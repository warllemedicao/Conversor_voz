from __future__ import annotations

import os
import re
import shutil
import subprocess
from urllib.request import urlopen
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/13uBrrPLx--PlHho0fcz5xW8eYehimRcC?usp=sharing"
CONFIG_FILE_ID = "1y_fKsgq8h_uWVCPDmzc9bnR2vmnJA1Pb"
TARGET_CHECKPOINT_NAME = "neuralepoch_2nd_00024.pth"
TARGET_CHECKPOINT_NAMES = (TARGET_CHECKPOINT_NAME, "epoch_2nd_00024.pth")
MODEL_ROOT = Path("/content/voz_neural")
OUTPUT_DIR = Path("/content/audios_gerados")
COLAB_DRIVE_ROOT = Path("/content/drive")


@dataclass(frozen=True)
class ModelBundle:
    engine: str
    model_path: Path
    config_path: Path | None = None


def run_command(command: list[str], input_text: str | None = None) -> None:
    subprocess.run(command, input=input_text, text=True, check=True)


def download_drive_folder(folder_url: str = DRIVE_FOLDER_URL, output_dir: Path = MODEL_ROOT) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_command(["gdown", "--folder", folder_url, "-O", str(output_dir), "--remaining-ok"])
    return output_dir


def download_config_file(file_id: str = CONFIG_FILE_ID, output_dir: Path = MODEL_ROOT) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "styletts2_config.yml"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        with urlopen(url, timeout=60) as response:
            config_path.write_bytes(response.read())
    except Exception:
        run_command(["gdown", "--id", file_id, "-O", str(config_path)])
    return config_path


def infer_model_project_root(checkpoint_path: Path) -> Path:
    for parent in checkpoint_path.parents:
        has_styletts_assets = (parent / "Utils").exists() or (parent / "Models").exists()
        if has_styletts_assets:
            return parent
    return checkpoint_path.parent


def copy_tree_contents(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_path in source_dir.rglob("*"):
        relative_path = source_path.relative_to(source_dir)
        target_path = output_dir / relative_path
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)


def copy_from_mounted_drive(
    drive_root: Path = COLAB_DRIVE_ROOT,
    output_dir: Path = MODEL_ROOT,
) -> bool:
    if not drive_root.exists():
        return False

    checkpoint_path = select_checkpoint_path(drive_root)
    if not checkpoint_path:
        return False

    source_root = infer_model_project_root(checkpoint_path)
    print(f"Checkpoint encontrado no Google Drive: {checkpoint_path}")
    print(f"Copiando arquivos de {source_root} para {output_dir}...")
    copy_tree_contents(source_root, output_dir)
    return True


def import_model_files(drive_root: Path = COLAB_DRIVE_ROOT, output_dir: Path = MODEL_ROOT) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_from_drive = copy_from_mounted_drive(drive_root=drive_root, output_dir=output_dir)
    if not copied_from_drive:
        print("Checkpoint nao encontrado no Drive montado. Baixando pasta publica com gdown...")
        download_drive_folder(output_dir=output_dir)

    print("Salvando YAML de configuracao dentro do Colab...")
    download_config_file(output_dir=output_dir)
    return output_dir


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def print_file_report(root: Path) -> None:
    files = list(iter_files(root))
    if not files:
        print("Nenhum arquivo encontrado.")
        return

    print("Arquivos encontrados:")
    for path in files:
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"- {path} ({size_mb:.1f} MB)")


def nearest_config_for_checkpoint(checkpoint: Path, root: Path, patterns: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.rglob(pattern))
    candidates = sorted(set(candidates))
    if not candidates:
        return None

    same_folder = [path for path in candidates if path.parent == checkpoint.parent]
    if same_folder:
        return same_folder[0]

    parent_folder = [path for path in candidates if checkpoint.parent.is_relative_to(path.parent)]
    if parent_folder:
        return sorted(parent_folder, key=lambda path: len(path.parts), reverse=True)[0]

    return candidates[0]


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch_2nd_(\d+)\.pth$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def select_checkpoint_path(root: Path) -> Path | None:
    pth_files = sorted(root.rglob("*.pth"))
    if not pth_files:
        return None

    for target_name in TARGET_CHECKPOINT_NAMES:
        matches = [path for path in pth_files if path.name == target_name]
        if matches:
            return sorted(matches)[0]

    epoch_files = [path for path in pth_files if checkpoint_epoch(path) >= 0]
    if epoch_files:
        return sorted(epoch_files, key=lambda path: (checkpoint_epoch(path), str(path)), reverse=True)[0]

    return pth_files[0]


def find_coqui_bundle(root: Path) -> ModelBundle | None:
    model_path = select_checkpoint_path(root)
    if not model_path:
        return None

    config_path = nearest_config_for_checkpoint(model_path, root, ("config.json",))
    if not config_path:
        return None

    return ModelBundle(engine="coqui", model_path=model_path, config_path=config_path)


def find_styletts2_bundle(root: Path) -> ModelBundle | None:
    model_path = select_checkpoint_path(root)
    if not model_path:
        return None

    config_path = nearest_config_for_checkpoint(model_path, root, ("*.yml", "*.yaml"))
    if not config_path:
        return None

    config_text = config_path.read_text(encoding="utf-8", errors="ignore")
    styletts2_markers = ("ASR_config", "PLBERT_dir", "model_params", "preprocess_params")
    if not all(marker in config_text for marker in styletts2_markers):
        return None

    return ModelBundle(engine="styletts2", model_path=model_path, config_path=config_path)


def find_piper_bundle(root: Path) -> ModelBundle | None:
    onnx_files = sorted(root.rglob("*.onnx"))
    if not onnx_files:
        return None
    return ModelBundle(engine="piper", model_path=onnx_files[0])


def detect_model_bundle(root: Path) -> ModelBundle:
    coqui = find_coqui_bundle(root)
    if coqui:
        return coqui

    styletts2 = find_styletts2_bundle(root)
    if styletts2:
        return styletts2

    piper = find_piper_bundle(root)
    if piper:
        return piper

    pth_files = sorted(root.rglob("*.pth"))
    if pth_files:
        names = ", ".join(path.name for path in pth_files)
        raise RuntimeError(
            "Encontrei arquivo .pth, mas nao encontrei uma configuracao suportada. "
            "Para usar checkpoint .pth como TTS, o Colab precisa da configuracao "
            f"da arquitetura do treinamento. Arquivos .pth encontrados: {names}"
        )

    raise RuntimeError(
        "Nao encontrei modelo suportado. Esperado: StyleTTS2 com .yml/.yaml + .pth, "
        "Coqui TTS com config.json + .pth, ou Piper com .onnx."
    )


def patch_torch_load_for_styletts2():
    import torch

    original_load = torch.load

    def load_with_legacy_checkpoint_support(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_legacy_checkpoint_support
    return original_load


def restore_torch_load(original_load) -> None:
    import torch

    torch.load = original_load


def report_styletts2_auxiliary_paths(config_path: Path) -> None:
    config_text = config_path.read_text(encoding="utf-8", errors="ignore")
    relative_paths = re.findall(r"(?:ASR_config|ASR_path|F0_path|PLBERT_dir):\s*([^,\n}]+)", config_text)
    missing_paths = []
    for relative_path in relative_paths:
        candidate = config_path.parent / relative_path.strip()
        if not candidate.exists():
            missing_paths.append(candidate)

    if missing_paths:
        print("Aviso: arquivos auxiliares do StyleTTS2 nao encontrados localmente.")
        print("O pacote styletts2 tentara baixar modelos padrao para esses itens:")
        for path in missing_paths:
            print(f"- {path}")


class NeuralVoiceSynthesizer:
    def __init__(self, bundle: ModelBundle, output_dir: Path = OUTPUT_DIR):
        self.bundle = bundle
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tts = None
        self._load()

    def _load(self) -> None:
        if self.bundle.engine == "coqui":
            import torch
            from TTS.api import TTS

            self.tts = TTS(
                model_path=str(self.bundle.model_path),
                config_path=str(self.bundle.config_path),
                progress_bar=True,
                gpu=torch.cuda.is_available(),
            )
            print("Modelo Coqui carregado.")
            print("Speakers:", getattr(self.tts, "speakers", None))
            print("Languages:", getattr(self.tts, "languages", None))
            return

        if self.bundle.engine == "piper":
            print("Modelo Piper pronto para inferencia.")
            return

        if self.bundle.engine == "styletts2":
            report_styletts2_auxiliary_paths(self.bundle.config_path)
            original_torch_load = patch_torch_load_for_styletts2()

            previous_cwd = Path.cwd()
            os.chdir(self.bundle.config_path.parent)
            try:
                from styletts2 import tts

                self.tts = tts.StyleTTS2(
                    model_checkpoint_path=str(self.bundle.model_path),
                    config_path=str(self.bundle.config_path),
                )
            finally:
                os.chdir(previous_cwd)
                restore_torch_load(original_torch_load)
            print("Modelo StyleTTS2 carregado.")
            return

        raise ValueError(f"Engine nao suportada: {self.bundle.engine}")

    @staticmethod
    def _choose_first(values, preferred: list[str] | None = None):
        if not values:
            return None
        preferred = preferred or []
        lowered = {str(value).lower(): value for value in values}
        for item in preferred:
            if item.lower() in lowered:
                return lowered[item.lower()]
        return values[0]

    def _next_wav_path(self) -> Path:
        index = len(list(self.output_dir.glob("audio_*.wav"))) + 1
        return self.output_dir / f"audio_{index:04d}.wav"

    def synthesize(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("Digite uma frase antes de gerar o audio.")

        wav_path = self._next_wav_path()
        if self.bundle.engine == "coqui":
            kwargs = {"text": text, "file_path": str(wav_path)}
            speakers = getattr(self.tts, "speakers", None)
            languages = getattr(self.tts, "languages", None)
            speaker = self._choose_first(speakers)
            language = self._choose_first(languages, ["pt-br", "pt", "portuguese", "portugues"])
            if speaker:
                kwargs["speaker"] = speaker
            if language:
                kwargs["language"] = language
            self.tts.tts_to_file(**kwargs)
            return str(wav_path)

        if self.bundle.engine == "styletts2":
            self.tts.inference(text, output_wav_file=str(wav_path), output_sample_rate=24000)
            return str(wav_path)

        run_command(["piper", "--model", str(self.bundle.model_path), "--output_file", str(wav_path)], input_text=text)
        return str(wav_path)


def create_gradio_app(synthesizer: NeuralVoiceSynthesizer):
    import gradio as gr

    def generate(text: str):
        try:
            audio_path = synthesizer.synthesize(text)
            return audio_path, audio_path, f"Audio gerado: {audio_path}"
        except Exception as exc:
            return None, None, f"Erro: {exc}"

    with gr.Blocks(title="Voz Neural") as demo:
        gr.Markdown("# Voz Neural\nDigite uma frase e pressione Enter para gerar um WAV.")
        text_box = gr.Textbox(label="Frase", placeholder="Digite sua frase e pressione Enter...", lines=1)
        audio = gr.Audio(label="Audio gerado", type="filepath")
        download = gr.File(label="Download do WAV")
        status = gr.Textbox(label="Status", interactive=False)
        text_box.submit(generate, inputs=text_box, outputs=[audio, download, status]).then(lambda: "", outputs=text_box)
    return demo


def main(import_files: bool = False) -> None:
    root = MODEL_ROOT
    if import_files:
        root = import_model_files()

    print_file_report(root)

    bundle = detect_model_bundle(root)
    print(f"Engine detectada: {bundle.engine}")
    print(f"Modelo: {bundle.model_path}")
    if bundle.config_path:
        print(f"Config: {bundle.config_path}")

    synthesizer = NeuralVoiceSynthesizer(bundle)
    demo = create_gradio_app(synthesizer)
    demo.launch(share=True, debug=True)


if __name__ == "__main__":
    main()
