from __future__ import annotations

import inspect
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# Patch para compatibilidade com NumPy 2.0+ em pacotes antigos (ex: scipy/nltk)
if not hasattr(np, "_no_nep50_warning"):
    setattr(np, "_no_nep50_warning", lambda: (lambda x: x))

# Patch para LangChain 0.2+ (styletts2 busca em langchain.text_splitter)
try:
    import langchain.text_splitter
except ImportError:
    try:
        import langchain_text_splitters
        import sys
        sys.modules["langchain.text_splitter"] = langchain_text_splitters
    except ImportError:
        pass


def patch_styletts2_typing():
    """Corrige TypeError: unsupported operand type(s) for /: 'str' and 'str' no PLBERT."""
    try:
        import styletts2.Utils.PLBERT.util
        from pathlib import Path

        original_load_plbert = styletts2.Utils.PLBERT.util.load_plbert

        def patched_load_plbert(log_dir, *args, **kwargs):
            if log_dir is not None and isinstance(log_dir, (str, bytes)):
                log_dir = Path(log_dir)
            return original_load_plbert(log_dir, *args, **kwargs)

        styletts2.Utils.PLBERT.util.load_plbert = patched_load_plbert
    except Exception:
        pass


def patch_styletts2_language():
    """Força o sotaque em português (PT-BR) no fonemizador Gruut."""
    try:
        from styletts2 import phoneme
        
        # O StyleTTS2 0.1.6 chama phonemize(text) sem passar o idioma, 
        # o que faz com que o Gruut use 'en-us' por padrão.
        # Aqui alteramos o padrão para 'pt-br'.
        original_phonemize = phoneme.GruutPhonemizer.phonemize
        
        def patched_phonemize(self, text, lang='pt-br'):
            return original_phonemize(self, text, lang=lang)
        
        phoneme.GruutPhonemizer.phonemize = patched_phonemize
        print("Sotaque StyleTTS2 configurado para Português (PT-BR).")
    except Exception as e:
        print(f"Aviso ao configurar idioma: {e}")


def fix_styletts2_config_paths(config_path: Path) -> None:
    """Transforma caminhos relativos no config.yml em caminhos absolutos recursivamente."""
    import yaml
    if not config_path.exists():
        return
    
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        base_dir = config_path.parent
        changed = False

        def walk_and_fix(data):
            nonlocal changed
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, (dict, list)):
                        walk_and_fix(v)
                    elif isinstance(v, str) and (k.endswith("_path") or k.endswith("_config") or k.endswith("_dir") or v.endswith(".pth") or v.endswith(".t7") or v.endswith(".yml")):
                        if v and not os.path.isabs(v):
                            # Tenta resolver relativo ao config.yml
                            abs_path = (base_dir / v).resolve()
                            if abs_path.exists():
                                data[k] = str(abs_path)
                                changed = True
            elif isinstance(data, list):
                for i in range(len(data)):
                    if isinstance(data[i], (dict, list)):
                        walk_and_fix(data[i])

        walk_and_fix(config)
        
        if changed:
            with config_path.open("w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Caminhos no {config_path.name} atualizados para absolutos.")
    except Exception as e:
        print(f"Aviso ao corrigir caminhos do config: {e}")


HF_REPO_ID = "warllem/Super_voz"
MODEL_ROOT = Path("/kaggle/working/Super_voz")
OUTPUT_DIR = Path("/kaggle/working/audios_gerados")
HF_METADATA_PATTERNS = (
    "model/*.yml",
    "model/*.yaml",
    "model/*.json",
    "model/*.txt",
    "model/Utils/**",
    "docs/**",
    "inference/**",
    "tokenizer/**",
    "data_reference/referencia_voz.wav",
    "data_reference/*.txt",
    "data_reference/*.csv",
)
REFERENCE_CANDIDATES = (
    "data_reference/referencia_voz.wav",
)


@dataclass(frozen=True)
class ModelBundle:
    engine: str
    model_path: Path
    config_path: Path | None = None
    reference_audio_path: Path | None = None


def run_command(command: list[str], input_text: str | None = None) -> None:
    subprocess.run(command, input=input_text, text=True, check=True)


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


def download_hf_repo(
    repo_id: str = HF_REPO_ID,
    output_dir: Path = MODEL_ROOT,
    token: str | None = None,
) -> Path:
    from huggingface_hub import snapshot_download, HfApi

    token = token or get_hf_token()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Baixar metadados e arquivos auxiliares (pequenos)
    print("Baixando metadados e arquivos auxiliares...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=HF_METADATA_PATTERNS,
        token=token,
    )
    
    # 2. Identificar qual o melhor checkpoint sem baixar todos
    print("Identificando checkpoint ideal...")
    api = HfApi(token=token)
    try:
        remote_files = api.list_repo_files(repo_id=repo_id)
    except Exception as e:
        print(f"Aviso ao listar arquivos remotos: {e}. Usando fallback best_model.pth")
        remote_files = ["model/best_model.pth"]

    pth_files = [f for f in remote_files if f.startswith("model/") and f.endswith(".pth") and "Utils" not in f]
    
    # Lógica de seleção (similar ao select_checkpoint_path mas para nomes remotos)
    selected_pth = "model/best_model.pth"
    
    best_metric = parse_key_value_file(output_dir / "model" / "best_metric.txt")
    if best_metric:
        source = best_metric.get("source_checkpoint")
        if source and f"model/{source}" in pth_files:
            selected_pth = f"model/{source}"
    
    if selected_pth not in pth_files and "model/latest_checkpoint.pth" in pth_files:
        selected_pth = "model/latest_checkpoint.pth"
    elif selected_pth not in pth_files and pth_files:
        selected_pth = sorted(pth_files)[-1] # Pega o último/mais recente se nada for achado

    # 3. Baixar apenas o arquivo selecionado
    if not (output_dir / selected_pth).exists():
        print(f"Baixando checkpoint selecionado: {selected_pth}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            allow_patterns=[selected_pth],
            token=token,
        )
    else:
        print(f"Checkpoint {selected_pth} já existe localmente.")

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


def parse_key_value_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def parse_validation_losses_from_log(log_path: Path) -> list[tuple[int, float]]:
    if not log_path.exists():
        return []

    results: list[tuple[int, float]] = []
    current_epoch = -1
    epoch_pattern = re.compile(r"Epoch\s+\[(\d+)/\d+\]")
    val_pattern = re.compile(r"Validation loss:\s*([0-9.]+)", re.IGNORECASE)

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        val_match = val_pattern.search(line)
        if val_match and current_epoch >= 0:
            results.append((current_epoch, float(val_match.group(1))))
    return results


def find_training_logs(root: Path) -> list[Path]:
    patterns = ("*.log", "*log*.txt", "events.out.tfevents.*")
    logs: list[Path] = []
    for pattern in patterns:
        logs.extend(root.rglob(pattern))
    return sorted(set(path for path in logs if path.is_file()))


def best_epoch_from_logs(root: Path) -> tuple[int, float] | None:
    best: tuple[int, float] | None = None
    preferred_logs = [path for path in (root / "docs" / "train.log", root / "model" / "train.log") if path.exists()]
    logs = preferred_logs or find_training_logs(root)
    for log_path in logs:
        if log_path.name.startswith("events.out.tfevents"):
            continue
        for epoch, loss in parse_validation_losses_from_log(log_path):
            if best is None or loss < best[1]:
                best = (epoch, loss)
    return best


def checkpoint_epoch(path: Path) -> int:
    patterns = (
        r"epoch_2nd_(\d+).*\.pth$",
        r"epoch_(\d+).*\.pth$",
    )
    for pattern in patterns:
        match = re.search(pattern, path.name)
        if match:
            return int(match.group(1))
    return -1


def select_checkpoint_path(root: Path) -> Path | None:
    pth_files = sorted(root.rglob("*.pth"))
    if not pth_files:
        return None

    best_metric = parse_key_value_file(root / "model" / "best_metric.txt")
    if best_metric:
        source_checkpoint = best_metric.get("source_checkpoint")
        candidates = [root / "model" / "best_model.pth"]
        if source_checkpoint:
            candidates.append(root / "model" / source_checkpoint)
        for candidate in candidates:
            if candidate.is_file():
                print("Melhor checkpoint pelo best_metric.txt:", candidate)
                print("Metricas:", best_metric)
                return candidate

    best_from_log = best_epoch_from_logs(root)
    if best_from_log:
        best_epoch, best_loss = best_from_log
        epoch_matches = [path for path in pth_files if checkpoint_epoch(path) == best_epoch]
        if epoch_matches:
            selected = sorted(epoch_matches)[0]
            print(f"Melhor checkpoint pelo train.log: {selected} (validation_loss={best_loss:.3f})")
            return selected

    latest_name_path = root / "model" / "latest_checkpoint.txt"
    if latest_name_path.exists():
        latest_name = latest_name_path.read_text(encoding="utf-8", errors="ignore").strip()
        for candidate in (root / "model" / "latest_checkpoint.pth", root / "model" / latest_name):
            if candidate.is_file():
                print("Checkpoint latest selecionado:", candidate)
                return candidate

    preferred_names = ("best_model.pth", "latest_checkpoint.pth")
    for name in preferred_names:
        matches = [path for path in pth_files if path.name == name]
        if matches:
            return sorted(matches)[0]

    epoch_files = [path for path in pth_files if checkpoint_epoch(path) >= 0]
    if epoch_files:
        return sorted(epoch_files, key=lambda path: (checkpoint_epoch(path), str(path)), reverse=True)[0]

    return pth_files[0]


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

    model_config = root / "model" / "config.yml"
    if model_config.exists() and model_config in candidates:
        return model_config
    return candidates[0]


def score_audio_reference_line(line: str) -> float | None:
    metric_patterns = (
        r"(?:score|similarity|mos|qualidade|quality)\s*[:=]\s*([0-9.]+)",
        r"(?:loss|erro|error|wer|cer)\s*[:=]\s*([0-9.]+)",
    )
    for pattern in metric_patterns:
        match = re.search(pattern, line, re.IGNORECASE)
        if not match:
            continue
        value = float(match.group(1))
        if re.search(r"loss|erro|error|wer|cer", pattern, re.IGNORECASE):
            value = -value
        return value
    return None


def select_reference_audio_from_logs(root: Path) -> Path | None:
    wav_pattern = re.compile(r"([\w./-]+\.wav)", re.IGNORECASE)
    best: tuple[float, Path] | None = None

    for log_path in find_training_logs(root):
        if log_path.name.startswith("events.out.tfevents"):
            continue
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            score = score_audio_reference_line(line)
            if score is None:
                continue
            for raw_path in wav_pattern.findall(line):
                candidate = (root / raw_path).resolve()
                if candidate.exists() and (best is None or score > best[0]):
                    best = (score, candidate)

    if best:
        print(f"Audio de referencia escolhido pelo log: {best[1]} (score={best[0]:.4f})")
        return best[1]
    return None


def select_reference_audio(root: Path) -> Path | None:
    from_log = select_reference_audio_from_logs(root)
    if from_log:
        return from_log

    for relative_path in REFERENCE_CANDIDATES:
        candidate = root / relative_path
        if candidate.exists():
            print("Audio de referencia escolhido:", candidate)
            return candidate

    val_list = root / "data_reference" / "val_list.txt"
    if val_list.exists():
        for line in val_list.read_text(encoding="utf-8", errors="ignore").splitlines():
            raw_path = line.split("|", 1)[0].strip()
            candidate = root / "data_reference" / raw_path
            if candidate.exists() and candidate.suffix.lower() == ".wav":
                print("Audio de referencia escolhido pela val_list:", candidate)
                return candidate

    wav_files = sorted((root / "data_reference").rglob("*.wav")) if (root / "data_reference").exists() else []
    if wav_files:
        print("Audio de referencia escolhido pelo primeiro WAV encontrado:", wav_files[0])
        return wav_files[0]

    return None


def find_coqui_bundle(root: Path) -> ModelBundle | None:
    model_path = select_checkpoint_path(root)
    if not model_path:
        return None

    config_path = nearest_config_for_checkpoint(model_path, root, ("config.json",))
    if not config_path:
        return None

    return ModelBundle("coqui", model_path, config_path, select_reference_audio(root))


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

    return ModelBundle("styletts2", model_path, config_path, select_reference_audio(root))


def find_piper_bundle(root: Path) -> ModelBundle | None:
    onnx_files = sorted(root.rglob("*.onnx"))
    if not onnx_files:
        return None
    return ModelBundle("piper", onnx_files[0], reference_audio_path=select_reference_audio(root))


def detect_model_bundle(root: Path) -> ModelBundle:
    styletts2 = find_styletts2_bundle(root)
    if styletts2:
        return styletts2

    coqui = find_coqui_bundle(root)
    if coqui:
        return coqui

    piper = find_piper_bundle(root)
    if piper:
        return piper

    pth_files = sorted(root.rglob("*.pth"))
    if pth_files:
        names = ", ".join(path.name for path in pth_files)
        raise RuntimeError(
            "Encontrei arquivo .pth, mas nao encontrei configuracao suportada. "
            f"Arquivos .pth encontrados: {names}"
        )

    raise RuntimeError("Nao encontrei modelo suportado em /kaggle/working/Super_voz.")


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
        print("Aviso: arquivos auxiliares do StyleTTS2 nao encontrados localmente:")
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
            return

        if self.bundle.engine == "piper":
            print("Modelo Piper pronto para inferencia.")
            return

        if self.bundle.engine == "styletts2":
            fix_styletts2_config_paths(self.bundle.config_path)
            report_styletts2_auxiliary_paths(self.bundle.config_path)
            patch_styletts2_typing()
            patch_styletts2_language()
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
            if self.bundle.reference_audio_path and "speaker_wav" in inspect.signature(self.tts.tts_to_file).parameters:
                kwargs["speaker_wav"] = str(self.bundle.reference_audio_path)
            self.tts.tts_to_file(**kwargs)
            return str(wav_path)

        if self.bundle.engine == "styletts2":
            kwargs = {"output_wav_file": str(wav_path), "output_sample_rate": 24000}
            parameters = inspect.signature(self.tts.inference).parameters
            
            for name in ("target_voice_path", "reference_audio_path", "speaker_wav"):
                if self.bundle.reference_audio_path and name in parameters:
                    kwargs[name] = str(self.bundle.reference_audio_path)
                    break
            self.tts.inference(text, **kwargs)
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

    with gr.Blocks(title="Super Voz") as demo:
        gr.Markdown("# Super Voz")
        text_box = gr.Textbox(label="Frase", placeholder="Digite sua frase e pressione Enter...", lines=1)
        audio = gr.Audio(label="Audio gerado", type="filepath")
        download = gr.File(label="Download do WAV")
        status = gr.Textbox(label="Status", interactive=False)
        text_box.submit(generate, inputs=text_box, outputs=[audio, download, status]).then(lambda: "", outputs=text_box)
    return demo


def display_notebook_audio(audio_path: str) -> str:
    from IPython.display import Audio, FileLink, display

    display(Audio(audio_path))
    display(FileLink(audio_path, result_html_prefix="Download do WAV: "))
    return audio_path


def synthesize_for_notebook(synthesizer: NeuralVoiceSynthesizer, text: str) -> str:
    audio_path = synthesizer.synthesize(text)
    print(f"Audio gerado: {audio_path}")
    return display_notebook_audio(audio_path)


def load_synthesizer(download: bool = False) -> NeuralVoiceSynthesizer:
    root = prepare_model_files(download=download)
    print_file_report(root)

    bundle = detect_model_bundle(root)
    print(f"Engine detectada: {bundle.engine}")
    print(f"Modelo: {bundle.model_path}")
    if bundle.config_path:
        print(f"Config: {bundle.config_path}")
    if bundle.reference_audio_path:
        print(f"Audio referencia: {bundle.reference_audio_path}")

    return NeuralVoiceSynthesizer(bundle)


def prepare_model_files(download: bool = True) -> Path:
    if download or not MODEL_ROOT.exists():
        return download_hf_repo()
    return MODEL_ROOT


def main(download: bool = False) -> None:
    synthesizer = load_synthesizer(download=download)
    demo = create_gradio_app(synthesizer)
    demo.launch(share=True, debug=True)


if __name__ == "__main__":
    main(download=True)
