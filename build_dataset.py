from attacker.dataset_builder import build_attack_dataset
import argparse
from typing import List, Optional
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_rewriter(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def _clean_prompt(text: str) -> str:
    text = text.strip()

    # 去掉常见 instruction leakage
    stop_markers = [
        "Note:",
        "Final Prompt:",
        "final prompt:",
        "Output:",
        "Assistant:",
        "User:",
        "Now generate",
        "ONE final prompt",
        "Do not",
        "Explanation:",
        "Reasoning:",
    ]

    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0].strip()

    # 只保留第一段
    text = text.split("\n\n")[0].strip()

    # 去掉 label
    prefixes = [
        "prompt:",
        "Prompt:",
        "Final prompt:",
        "final prompt:",
        ":",
    ]

    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()

    # 最多保留前两句
    sentences = re.split(r'(?<=[.!?])\s+', text)

    if len(sentences) > 2:
        text = " ".join(sentences[:2])

    return text.strip()


def _generate_prompts(
    prompts: List[str],
    model_path: str,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:

    tokenizer, model = _load_rewriter(model_path, device)

    generated_results = []

    with torch.no_grad():

        for start in range(0, len(prompts), batch_size):

            chunk = prompts[start:start + batch_size]

            chat_inputs = []

            for prompt in chunk:

                messages = [
                    {
                        "role": "system",
                        "content":
                        (
                            "You generate exactly ONE natural "
                            "conversational prompt.\n"
                            "Return ONLY the final prompt.\n"
                            "No explanation.\n"
                            "No notes.\n"
                            "No labels.\n"
                            "No metadata."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                chat_inputs.append(text)

            encoded = tokenizer(
                chat_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            encoded = {
                k: v.to(device)
                for k, v in encoded.items()
            }

            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.15,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

            input_lengths = encoded["attention_mask"].sum(dim=1)

            for i in range(len(chunk)):

                generated_ids = outputs[i][input_lengths[i]:]

                text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True
                ).strip()

                text = _clean_prompt(text)

                generated_results.append(text)

    return generated_results

def _generate_prompts_with_vllm(
    prompts: List[str],
    model_path: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    gpu_mem_util: float,
    max_model_len: Optional[int],
) -> List[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    chat_prompts = []

    for prompt in prompts:

        messages = [
            {
                "role": "system",
                "content":
                (
                    "You generate exactly ONE natural "
                    "conversational prompt.\n"
                    "Return ONLY the final prompt.\n"
                    "No explanation.\n"
                    "No notes.\n"
                    "No labels.\n"
                    "No metadata."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        chat_prompts.append(text)

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        repetition_penalty=1.15,
        stop=[
            "\n\n",
            "Note:",
            "Assistant:",
            "Final Prompt:",
            "Output:",
        ]
    )

    outputs = llm.generate(chat_prompts, params)

    results = []

    for output in outputs:
        if output.outputs:
            text = output.outputs[0].text.strip()
            text = _clean_prompt(text)
            results.append(text)
        else:
            results.append("")

    return results

# 现在有逻辑错误，不需要将生成攻击语句的prompt进行rewrite了，直接用base_prompt当作训练集，或者加入一些优化使其更复杂。原来的rewrite逻辑放到另外的测试文件中，测试这些生成攻击语句prompt生成攻击语句的效果

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="/home/fangzibang/SelfEvolvingPrivacyRL/data")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--rewrite_backend",
        type=str,
        choices=["transformers", "vllm"],
        default="transformers",
    )
    parser.add_argument("--vllm_gpu_mem_util", type=float, default=0.8)
    parser.add_argument("--vllm_max_model_len", type=int, default=8192)
    args = parser.parse_args()

    build_attack_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        rewrite_backend=args.rewrite_backend,
        vllm_gpu_mem_util=args.vllm_gpu_mem_util,
        vllm_max_model_len=args.vllm_max_model_len,
    )
