import os
import threading
import time
from typing import Dict, Tuple
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_SYSTEM_PROMPT = (
    "You are a specialized domain expert. Analyze the provided input accurately "
    "and produce a structured, technically valid response following domain standards."
)


class DualModeExpertEngine:
    """
    Loads the 4-bit quantized base model ONCE into VRAM and dynamically toggles
    the QLoRA adapter off/on for side-by-side Before vs. After comparisons.
    """

    def __init__(self, base_model_id: str, adapter_dir: str):
        if not os.path.exists(adapter_dir):
            raise FileNotFoundError(
                f"LoRA adapter directory not found at '{adapter_dir}'. "
                "Verify your Google Drive mount or adapter path."
            )

        self.base_model_id = base_model_id
        self.adapter_dir = adapter_dir
        self._lock = threading.Lock()

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
        self.model.eval()

    def _generate_single(
        self,
        prompt_text: str,
        max_new_tokens: int,
        temperature: float,
    ) -> Tuple[str, Dict[str, float]]:
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        do_sample = temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        start_t = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        elapsed = max(time.perf_counter() - start_t, 1e-6)

        generated_ids = output_ids[0][input_len:]
        num_tokens = len(generated_ids)
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        metrics = {
            "latency_sec": round(elapsed, 2),
            "tokens": num_tokens,
            "tokens_per_sec": round(num_tokens / elapsed, 1),
        }
        return response_text, metrics

    def compare(
        self,
        user_input: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Dict:
        """Runs both Base Model (adapter disabled) and QLoRA Expert (adapter enabled)."""
        if not user_input or not user_input.strip():
            raise ValueError("Input text cannot be empty.")

        messages = [
            {"role": "system", "content": system_prompt.strip() or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_input.strip()},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        with self._lock:
            # 1. Base Model pass (LoRA adapter bypassed)
            with self.model.disable_adapter():
                base_output, base_metrics = self._generate_single(
                    prompt_text, max_new_tokens, temperature
                )

            # 2. Fine-Tuned QLoRA pass (LoRA adapter active)
            expert_output, expert_metrics = self._generate_single(
                prompt_text, max_new_tokens, temperature
            )

        vram_gb = (
            round(torch.cuda.max_memory_allocated() / (1024**3), 2)
            if torch.cuda.is_available()
            else 0.0
        )

        return {
            "base_output": base_output,
            "base_metrics": base_metrics,
            "expert_output": expert_output,
            "expert_metrics": expert_metrics,
            "peak_vram_gb": vram_gb,
        }