import os
import json
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main():
    parser = argparse.ArgumentParser(
        description="Process MMLU JSON files with a LoRA-enhanced LLM"
    )
    parser.add_argument(
        "--directory_path",
        type=str,
        default=r"path/mmlu_data",
        help="Path to the directory containing JSON files"
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        default="path/output.json",
        help="Path to the output JSON file"
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="path/model",
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=r"path/lora_model",
        help="Path to the LoRA weights"
    )
    args = parser.parse_args()

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_path,
        use_fast=True,
        trust_remote_code=True
    )

    # Initialize the LLM with LoRA support
    llm = LLM(
        model=args.pretrained_model_path,
        enable_lora=True,
        max_model_len=2048,
        tensor_parallel_size=1,
        trust_remote_code=True
    )

    # Configure LoRA
    lora_request = LoRARequest(
        "duan_lora",
        rank=1,
        lora_path=args.lora_path
    )

    # Default sampling parameters
    sampling_params = SamplingParams(max_tokens=2048)

    # Iterate through JSON files in the directory
    for filename in os.listdir(args.directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(args.directory_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                datas = json.load(file)

            test_prompts = []
            for data in datas:
                question = data["question"]
                options = data["options"]
                subject = data["subject"]

                # Construct the prompt for a multiple-choice question
                prompt = (
                    f"The following are multiple-choice questions about knowledge about {subject}. "
                    f"Please choose the correct answer.\n{question}\n{options}"
                )
                messages = [{"role": "user", "content": prompt}]

                # Tokenize the chat template and prepare for generation
                query = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                test_prompts.append(query)

            # Generate model outputs
            outputs = llm.generate(
                test_prompts,
                sampling_params,
                lora_request=lora_request
            )

            for data, output in zip(datas, outputs):
                response = output.outputs[0].text
                print(response)
                data["model_output"] = response
                print("-" * 50)

            # Save the updated data to the output JSON file
            with open(args.output_file_path, "w", encoding="utf-8") as outfile:
                json.dump(datas, outfile, ensure_ascii=False, indent=4)

    print("Processing complete.")


if __name__ == "__main__":
    main()
