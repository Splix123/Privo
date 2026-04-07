from llama_cpp import Llama


class LocalLLM:
    def __init__(
        self,
        model_path: str = "models/llm/qwen2.5-3b-instruct-q4_k_m.gguf",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.5,
        history_limit: int = 6,
    ) -> None:
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

    def generate(self, user_text: str, system_prompt: str | None = None) -> str:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})

        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response["choices"][0]["message"]["content"].strip()
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": answer})
        max_messages = self.history_limit * 2
        self.history = self.history[-max_messages:]

        # TODO: LLM Stream?
        return answer

    def reset_history(self) -> None:
        self.history = []

    # --- DEL ---
    # n_ctx, n_gpu_layers, verbose, max_tokens, temperature als Parameter in generate
