from rich.console import Console
from privo.app.config_loader import ConfigLoader
from privo.app.debugger import Debugger
from privo.audio import AudioInput
from privo.wakeword import WakewordDetector
from privo.stt import UtteranceRecorder
from privo.stt import WhisperStt
from privo.llm import LocalLLM
from privo.tts import PiperTts

class ModuleBuilder:
    def __init__(self, debug: bool = False):
        self.console = Console()
        self.config = ConfigLoader().load()
        self.debugger = Debugger(debug_dir=self.config.get("debug_dir", "debug"), enabled=debug)

    def build_audio(self) -> AudioInput:
        audio = AudioInput(
            **{
                k: v for k, v in {
                    "sample_rate": self.config.get("au_sample_rate"),
                    "block_ms": self.config.get("au_block_size"),
                    "channels": self.config.get("au_channels"),
                    "ring_buffer_chunks": self.config.get("au_ring_buffer_chunks"),
                }.items()
                if v is not None
            }
        )
        self.console.print("Audio-Modul geladen")
        return audio

    def build_wakeword_detector(self) -> WakewordDetector:
        detector = WakewordDetector(
            **{
                k: v for k, v in {
                    "model_path": self.config.get("wwd_model_path"),
                    "threshold": self.config.get("wwd_threshold"),
                    "vad_threshold": self.config.get("wwd_vad_threshold"),
                }.items()
                if v is not None
            }
        )
        self.console.print("Wakeword-Detector geladen")
        return detector

    def build_recorder(self) -> UtteranceRecorder:
        recorder = UtteranceRecorder(
            **{
                k: v for k, v in {
                    "silence_threshold": self.config.get("stt_silence_threshold"),
                    "silence_blocks": self.config.get("stt_silence_blocks"),
                }.items()
                if v is not None
            }
        )
        self.console.print("Utterance-Recorder-Modul geladen")
        return recorder
    
    def build_stt(self) -> WhisperStt:
        stt = WhisperStt(
            **{
                k: v for k, v in {
                    "model_path": self.config.get("stt_model_path"),
                    "device": self.config.get("stt_device"),
                    "compute_type": self.config.get("stt_compute_type"),
                    "language": self.config.get("stt_language"),
                    "beam_size": self.config.get("stt_beam_size"),
                }.items()
                if v is not None
            }
        )
        self.console.print("STT-Modul geladen")
        return stt
    def build_llm(self) -> LocalLLM:
        llm = LocalLLM(
            **{
                k: v for k, v in {
                    "model_path": self.config.get("llm_model_path"),
                    "n_ctx": self.config.get("llm_n_ctx"),
                    "n_gpu_layers": self.config.get("llm_n_gpu_layers"),
                    "verbose": self.config.get("llm_verbose"),
                    "max_tokens": self.config.get("llm_max_tokens"),
                    "temperature": self.config.get("llm_temperature"),
                    "history_limit": self.config.get("llm_history_limit"),
                }.items()
                if v is not None
            }
        )
        self.console.print("LLM-Modul geladen")
        return llm

    def build_tts(self) -> PiperTts:
        tts = PiperTts(
            **{
                k: v for k, v in {
                    "model_path": self.config.get("tts_model_path"),
                    "config_path": self.config.get("tts_config_path"),
                    "speaker": self.config.get("tts_speaker"),
                    "length_scale": self.config.get("tts_length_scale"),
                    "noise_scale": self.config.get("tts_noise_scale"),
                    "noise_w_scale": self.config.get("tts_noise_w_scale"),
                    "sentence_silence": self.config.get("tts_sentence_silence"),
                }.items()
                if v is not None
            }
        )
        self.console.print("TTS-Modul geladen\n")
        return tts
    
    def build_all(self):
        config = self.config
        debugger = self.debugger
        audio = self.build_audio()
        detector = self.build_wakeword_detector()
        recorder = self.build_recorder()
        stt = self.build_stt()
        llm = self.build_llm()
        tts = self.build_tts()

        return config, debugger, audio, detector, recorder, stt, llm, tts
    
    def build_benchmark(self):
        config = self.config
        debugger = self.debugger
        stt = self.build_stt()
        llm = self.build_llm()
        tts = self.build_tts()

        return config, debugger, stt, llm, tts