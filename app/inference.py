import os
import threading
import time
import json
from typing import Any, Dict, Tuple

try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:  # The API can still be imported for CPU tests.
    torch = None
    PeftModel = AutoModelForCausalLM = AutoTokenizer = BitsAndBytesConfig = None


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


class TriageInferenceEngine:
    """Production triage engine with vLLM and single-GPU Transformers backends.

    Model loading is explicit and happens only when this class is instantiated.
    This keeps FastAPI imports lightweight and allows mocked engines in CI.
    """

    def __init__(
        self,
        model_id: str | None = None,
        default_adapter_path: str | None = None,
        backend: str | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        self.model_id = model_id or os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
        self.default_adapter_path = default_adapter_path or os.getenv(
            "DEFAULT_ADAPTER_PATH", "adapters/final_adapter"
        )
        self.backend = (backend or os.getenv("INFERENCE_BACKEND", "transformers")).lower()
        self.confidence_threshold = confidence_threshold or float(
            os.getenv("CONFIDENCE_THRESHOLD", "0.85")
        )
        self.adapter_version = os.getenv("ADAPTER_VERSION", "final")
        self._lock = threading.RLock()
        self.model: Any = None
        self.tokenizer: Any = None
        self._adapters: set[str] = set()
        self._load_model()

    def _load_model(self) -> None:
        if self.backend == "vllm":
            try:
                from vllm import LLM
            except ImportError as exc:
                raise RuntimeError("INFERENCE_BACKEND=vllm requires vllm") from exc
            self.model = LLM(
                model=self.model_id,
                enable_lora=True,
                max_lora_rank=64,
                trust_remote_code=True,
            )
            return
        if self.backend != "transformers":
            raise ValueError("INFERENCE_BACKEND must be 'vllm' or 'transformers'")
        if AutoTokenizer is None or torch is None:
            raise RuntimeError("Transformers inference dependencies are not installed")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            self.model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True
        )
        if self.default_adapter_path and os.path.exists(self.default_adapter_path):
            self.model = PeftModel.from_pretrained(
                base, self.default_adapter_path, is_trainable=False
            )
            self._adapters.add(self.default_adapter_path)
        else:
            self.model = base
        self.model.eval()

    def enable_adapter(self, adapter_path: str) -> None:
        """Load an adapter into the existing base model and make it active."""
        with self._lock:
            if self.backend == "vllm":
                self._adapters.add(adapter_path)
                return
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
            if adapter_path not in self._adapters:
                if not hasattr(self.model, "load_adapter"):
                    raise RuntimeError("The loaded model does not support LoRA adapters")
                self.model.load_adapter(adapter_path, adapter_name=adapter_path)
                self._adapters.add(adapter_path)
            self.set_adapter(adapter_path)

    def set_adapter(self, adapter_path: str) -> None:
        """Switch the active LoRA adapter without reloading base weights."""
        with self._lock:
            if self.backend != "vllm" and hasattr(self.model, "set_adapter"):
                self.model.set_adapter(adapter_path)
            self.adapter_version = os.path.basename(adapter_path.rstrip("/\\"))

    @staticmethod
    def _confidence_from_scores(scores: Any) -> float:
        if not scores:
            return 0.0
        values = []
        for score in scores:
            probabilities = torch.softmax(score[0].float(), dim=-1)
            values.append(float(probabilities.max().item()))
        return max(0.0, min(1.0, sum(values) / len(values)))

    def _transformers_predict(self, prompt: str) -> tuple[dict[str, Any], float]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output.sequences[0][inputs["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            return json.loads(raw), self._confidence_from_scores(output.scores)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("Model returned invalid triage JSON") from exc

    def _vllm_predict(self, prompt: str, adapter_path: str | None) -> tuple[dict[str, Any], float]:
        from app.schemas import TriagePrediction
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        schema = TriagePrediction.model_json_schema()
        params = SamplingParams(
            temperature=0,
            max_tokens=192,
            guided_decoding=GuidedDecodingParams(json=schema),
            logprobs=1,
        )
        lora = None
        if adapter_path:
            from vllm.lora.request import LoRARequest

            lora = LoRARequest("tenant-adapter", 1, adapter_path)
        result = self.model.generate([prompt], params, lora_request=lora)[0]
        candidate = result.outputs[0]
        return json.loads(candidate.text), self._confidence_from_scores(
            [torch.tensor([list(item.values())[0]]) for item in (candidate.logprobs or [])]
        ) if torch is not None else 0.0

    def predict(self, ticket_text: str, adapter_path: str | None = None) -> tuple[Any, float]:
        """Classify a ticket and return validated labels plus confidence."""
        from app.schemas import TriagePrediction
        from app.security import build_inference_prompt

        prompt = build_inference_prompt(ticket_text)
        with self._lock:
            if adapter_path:
                self.enable_adapter(adapter_path)
            payload, confidence = (
                self._vllm_predict(prompt, adapter_path)
                if self.backend == "vllm"
                else self._transformers_predict(prompt)
            )
        prediction = TriagePrediction.model_validate(payload)
        return prediction, confidence