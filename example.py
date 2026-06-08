import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 1. 根据Qwen3-0.6B文件夹内容初始化Tokenizer以及LLM模型
    path = os.environ.get(
        "NANOVLLM_MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "huggingface", "Qwen3-0.6B"),
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
