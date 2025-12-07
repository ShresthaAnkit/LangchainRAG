from pathlib import Path
from typing import Dict


class PromptManager:
    def __init__(self):
        self.prompts_dir = Path("app", "prompt")
        self._cache: Dict[str, str] = {}

    def get_prompt(self, name: str) -> str:
        """
        Get a prompt by name (without .txt). Loads from file on first request.
        """
        if name in self._cache:
            return self._cache[name]

        prompt_file = self.prompts_dir / f"{name}.txt"
        if not prompt_file.exists():
            raise ValueError(f"Prompt '{name}' not found in {self.prompts_dir}")

        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()

        self._cache[name] = prompt_text
        return prompt_text

    def list_prompts(self) -> list[str]:
        """
        List all available prompt names (without .txt extension).
        """
        return [f.stem for f in self.prompts_dir.glob("*.txt")]

prompt_manager = PromptManager()