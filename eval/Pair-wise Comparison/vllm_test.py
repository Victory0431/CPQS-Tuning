import os
import json
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm

# Parse command-line arguments for directory and model paths
def get_args():
    parser = argparse.ArgumentParser(
        description="Generate model outputs for JSON test data using LoRA-enabled LLM"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"path/test_data",
        help="Directory containing input JSON files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"path/output",
        help="Directory where output JSON files will be saved"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=r"path/model",
        help="Path to the pretrained model directory"
    )
    return parser.parse_args()

# Retrieve arguments
args = get_args()

# Assign paths from parsed arguments
input_directory_path = args.input_dir
output_directory_path = args.output_dir
pretrained_model_pth = args.model_path

# Ensure that the output directory exists
os.makedirs(output_directory_path, exist_ok=True)

# Initialize the tokenizer with the specified pretrained model path
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_pth,
    use_fast=True,
    trust_remote_code=True
)

# Initialize the LLM with LoRA support and bfloat16 precision
tokens_llm = LLM(
    model=pretrained_model_pth,
    dtype="bfloat16",
)

# Define sampling parameters: maximum tokens and deterministic output
sampling_params = SamplingParams(max_tokens=1024, temperature=0)

# Counters for correct and error cases across all files
total_correct = 0
total_errors = 0

# Iterate over all JSON files in the input directory
for filename in os.listdir(input_directory_path):
    if filename.endswith('.json'):
        # Build full file paths for input and output
        input_file_path = os.path.join(input_directory_path, filename)
        output_file_path = os.path.join(output_directory_path, filename)

        # Load JSON data from the input file
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            datas = json.load(infile)

        # Prepare prompts by applying the chat template to each record
        test_prompts = []
        for item in datas:
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            question = f"{instruction} Input: {input_text}"
            messages = [{"role": "user", "content": question}]
            query = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            test_prompts.append(query)

        # Generate outputs using the LoRA-enabled LLM
        outputs = llm.generate(test_prompts, sampling_params)

        # Attach the model's responses back to each record and print them
        for item, output in zip(datas, outputs):
            response = output.outputs[0].text
            print(response)
            item['model_output'] = response
            print('--------------------------------------------------')

        # Save the updated data with model outputs to the output file
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(datas, outfile, ensure_ascii=False, indent=4)

print("All files have been processed successfully.")
