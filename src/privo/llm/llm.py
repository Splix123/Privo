from pathlib import Path
from llama_cpp import Llama


class LocalLLM:
    """LLM-Modul (llama.cpp)"""

    def __init__(
        self,
        model_path: str = "models/llm/qwen2.5-3b-instruct-q4_k_m.gguf",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.5,
        history_limit: int = 6,
        system_prompt: str | None = None,
    ) -> None:
        """Initialisiert das LLM-Modul mit Parametern aus der config.yaml oder den gegebenen Standardwerten.

        Args:
            model_path (str, optional): Pfad zum LLM-Modell. Defaults to "models/llm/qwen2.5-3b-instruct-q4_k_m.gguf".
            n_ctx (int, optional): Contextfenster. Defaults to 4096.
            n_gpu_layers (int, optional): Aktiviert GPU-Acceleration bei -1. Defaults to -1.
            verbose (bool, optional): Aktiviert ausführliche Ausgabe. Defaults to False.
            max_tokens (int, optional): Maximale Anzahl an Tokens. Defaults to 256.
            temperature (float, optional): Temperatur für die Textgenerierung. Defaults to 0.5.
            history_limit (int, optional): Limit für die Gesprächshistorie. Defaults to 6.
            system_prompt (str | None, optional): System-Prompt für das LLM. Defaults to None.

        Raises:
            FileNotFoundError: Wenn das LLM-Modell nicht gefunden wird.
            ValueError: Wenn n_ctx kleiner oder gleich 0 ist.
            ValueError: Wenn max_tokens kleiner oder gleich 0 ist.
            ValueError: Wenn history_limit kleiner als 0 ist.
            ValueError: Wenn temperature kleiner als 0 ist.
        """
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"LLM-Modell nicht gefunden: {model_file}")
        if n_ctx <= 0:
            raise ValueError("n_ctx muss größer als 0 sein")
        if max_tokens <= 0:
            raise ValueError("max_tokens muss größer als 0 sein")
        if history_limit < 0:
            raise ValueError("history_limit muss größer oder gleich 0 sein")
        if temperature < 0:
            raise ValueError("temperature muss größer oder gleich 0 sein")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history_limit = history_limit
        self.history: list[dict[str, str]] = []
        self.system_prompt = system_prompt

    def generate(self, user_text: str) -> str:
        """Generiert eine Antwort basierend auf dem Benutzereingabetext und der Gesprächshistorie.

        Args:
            user_text (str): Der Text, den der Benutzer eingegeben / Gesprochen hat.

        Raises:
            RuntimeError: Wenn ein Fehler bei der Textgenerierung auftritt.

        Returns:
            str: Die generierte Antwort des LLM.
        """
        user_text = user_text.strip()
        if not user_text:
            return ""

        messages = [{"role": "system", "content": self.system_prompt}]

        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})

        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"Fehler bei der Textgenerierung: {e}") from e

        content = response["choices"][0]["message"].get("content", "")
        answer = content.strip() if content else ""

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": answer})

        max_messages = self.history_limit * 2
        self.history = self.history[-max_messages:]

        return answer

    def reset_history(self) -> None:
        """Setzt die History (also das Gedächtnis) der Konversation zurück."""
        self.history.clear()

    # --- DEL ---
    # n_ctx, n_gpu_layers, verbose, max_tokens, temperature als Parameter in generate
