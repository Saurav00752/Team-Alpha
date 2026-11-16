import argparse
import json
import os
import gradio as gr
import yaml
from inference import DEFAULT_SYSTEM_PROMPT, DualModeExpertEngine


def load_preset_examples(test_file: str, max_examples: int = 4) -> list:
    """Loads held-out samples from test.jsonl as one-click demo presets."""
    if not os.path.exists(test_file):
        return []
    presets = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            user_msg = next((m["content"] for m in row["messages"] if m["role"] == "user"), None)
            if user_msg:
                presets.append([user_msg])
            if len(presets) >= max_examples:
                break
    return presets


def build_demo(engine: DualModeExpertEngine, test_file: str) -> gr.Blocks:
    presets = load_preset_examples(test_file)

    def run_comparison(user_input: str, system_prompt: str, max_tokens: int, temperature: float):
        if not user_input or not user_input.strip():
            return (
                "Please enter a domain query or select a preset example below.",
                "",
                "Status: Waiting for input",
            )
        try:
            res = engine.compare(
                user_input=user_input,
                system_prompt=system_prompt,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
            )
            b_m = res["base_metrics"]
            e_m = res["expert_metrics"]

            base_header = (
                f"{res['base_output']}\n\n"
                f"---\n"
                f"Latency: {b_m['latency_sec']}s | Speed: {b_m['tokens_per_sec']} tok/s | Tokens: {b_m['tokens']}"
            )
            expert_header = (
                f"{res['expert_output']}\n\n"
                f"---\n"
                f"Latency: {e_m['latency_sec']}s | Speed: {e_m['tokens_per_sec']} tok/s | Tokens: {e_m['tokens']}"
            )
            telemetry = (
                f"**Active Base Model:** `{engine.base_model_id}` | "
                f"**Peak GPU VRAM (Single-Load Shared Weights):** `{res['peak_vram_gb']} GB`"
            )
            return base_header, expert_header, telemetry
        except Exception as exc:
            return f"Error: {exc}", f"Error: {exc}", "Status: Inference error"

    with gr.Blocks(title="Domain-Expert QLoRA Comparator", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Domain-Expert LLM: Before vs. After QLoRA Fine-Tuning\n"
            "Compare the **un-tuned 4-bit Base Model** against our **QLoRA Domain Expert** "
            "side by side on identical prompts using a single shared GPU memory footprint."
        )

        with gr.Row():
            user_box = gr.Textbox(
                label="Domain Input (Contract Clause, Raw Technical Log, or Medical Record)",
                placeholder="Paste a raw domain input here or click a held-out test preset below...",
                lines=6,
            )

        with gr.Accordion("Advanced Generation & System Prompt Settings", open=False):
            sys_box = gr.Textbox(label="System Instruction", value=DEFAULT_SYSTEM_PROMPT, lines=2)
            with gr.Row():
                max_tok_slider = gr.Slider(
                    minimum=64, maximum=1024, value=256, step=32, label="Max New Tokens"
                )
                temp_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Temperature (0 = Deterministic)"
                )

        run_btn = gr.Button("Compare Base Model vs. QLoRA Expert", variant="primary")
        telemetry_md = gr.Markdown(f"**Loaded Base Model:** `{engine.base_model_id}`")

        with gr.Row():
            base_out = gr.Textbox(
                label="Before: General Base Model (Adapter Disabled)",
                lines=14,
                interactive=False,
            )
            expert_out = gr.Textbox(
                label="After: Domain-Expert Model (QLoRA Adapter Active)",
                lines=14,
                interactive=False,
            )

        if presets:
            gr.Examples(
                examples=presets,
                inputs=[user_box],
                label="Held-Out Golden Test Presets (Zero Training Overlap)",
            )

        run_btn.click(
            fn=run_comparison,
            inputs=[user_box, sys_box, max_tok_slider, temp_slider],
            outputs=[base_out, expert_out, telemetry_md],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Side-by-Side QLoRA Comparison Demo")
    parser.add_argument("--config", type=str, default="training/configs/config.yaml")
    parser.add_argument("--adapter-dir", type=str, default=None)
    parser.add_argument("--test-file", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link (for Colab)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    adapter_path = args.adapter_dir or cfg["training"]["final_adapter_dir"]
    engine = DualModeExpertEngine(
        base_model_id=cfg["model"]["base_model_id"],
        adapter_dir=adapter_path,
    )

    demo = build_demo(engine, test_file=args.test_file)
    demo.launch(share=args.share or ("COLAB_RELEASE_TAG" in os.environ))