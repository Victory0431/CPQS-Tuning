"""
Batch-evaluate two sets of JSON files with GPT-4o.

Only change from the previous version:
‣ The three path constants are now command-line arguments handled by argparse,
  each with the same default value as before.

Run, e.g.:
python eval_pairs.py \
  --input_dir_1 "V:\\custom\\first_batch" \
  --input_dir_2 "V:\\custom\\second_batch" \
  --output_dir  "V:\\results"
"""
import os
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import json

# Initialize OpenAI client (please use a secure method to manage your API key)
client = OpenAI(api_key="")  # Please replace with your API key

# Paths for input and output folders
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate two assistants’ answers with GPT-4o."
    )
    parser.add_argument(
        "--input_dir_1",
        type=str,
        default=r"path/model1_output",
        help="First batch of JSON files",
    )
    parser.add_argument(
        "--input_dir_2",
        type=str,
        default=r"path/model2_output",
        help="Second batch of JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"path/output",
        help="Folder to write merged results",
    )
    return parser.parse_args()


args = get_args()

input_dir_1: str = args.input_dir_1
input_dir_2: str = args.input_dir_2
output_dir:  str = args.output_dir
os.makedirs(output_dir, exist_ok=True)

criteria = (
    "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\n"
    "Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\n"
    "Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
)

def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = [s.strip() for s in score_pair.split(" ") if s.strip() != ""]
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        print(f"{e}\nContent: {review}\nYou must manually fix the score pair.")
        return [-1, -1]

def call_api(question, answer_1, answer_2):
    prompt_template = (
        f"[Question]\n{question}\n\n"
        f"[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n"
        f"[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n"
        f"[System]\n{criteria}\n\n"
    )
    print("Sending prompt to OpenAI API:")
    print(prompt_template)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                {"role": "user", "content": prompt_template}
            ],
            temperature=0.0,
            top_p=1.0
        )
        review = response.choices[0].message.content
        print("Received review from OpenAI API:")
        print(review)
        scores = parse_score(review)
        return review, scores
    except Exception as e:
        print(f"OpenAI API 调用失败: {e}")
        return "", [-1, -1]

# Retrieve all JSON filenames (without paths) from both folders
files_1 = {os.path.basename(f) for f in glob.glob(os.path.join(input_dir_1, "*.json"))}
files_2 = {os.path.basename(f) for f in glob.glob(os.path.join(input_dir_2, "*.json"))}

# Find common files in both folders
common_files = files_1.intersection(files_2)

for filename in common_files:
    file_path_1 = os.path.join(input_dir_1, filename)
    file_path_2 = os.path.join(input_dir_2, filename)

    output_file = os.path.join(output_dir, filename)

    # Load existing output file if it exists; otherwise load from input files
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as pf:
            data = json.load(pf)
    else:
        with open(file_path_1, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        with open(file_path_2, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
        
        # Add data2's 'model_output' to data1 as 'model_output2'
        for item1, item2 in zip(data1, data2):
            item1['model_output2'] = item2.get('model_output', '')
        
        data = {
            "data1": data1
        }

    # Build an index for data1 by instruction
    data1_by_instruction = {item['instruction']: item for item in data['data1']}

    # Get the set of instructions in data1
    data1_instructions = set(item['instruction'] for item in data['data1'])

    # Identify entries that need processing (missing scores)
    tasks = []
    instruction_to_index_1 = {e['instruction']: i for i, e in enumerate(data['data1'])}

    for entry1 in data['data1']:
        instruction = entry1.get('instruction', '')
        if instruction in data1_instructions:
            has_score_normal = ('answer_1_score' in entry1) and ('answer_2_score' in entry1)
            has_score_reverse = ('answer_1_score_reverse' in entry1) and ('answer_2_score_reverse' in entry1)

            # Only process entries that don't have scores yet
            if not has_score_normal or not has_score_reverse:
                input_text = entry1.get('input', '')
                question = f"{instruction} Input: {input_text}"
                answer_1 = entry1.get('model_output', '')
                answer_2 = entry1.get('model_output2', '')
                tasks.append((instruction, question, answer_1, answer_2))

    if not tasks:
        # No entries to process; write back current data to ensure it's up to date
        with open(output_file, 'w', encoding='utf-8') as out:
            json.dump(data, out, ensure_ascii=False, indent=4)
        continue

    # Process tasks in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_task = {}
        for (instr, q, a1, a2) in tasks:
            # Normal order evaluation
            if not ('answer_1_score' in data['data1'][instruction_to_index_1[instr]] and
                    'answer_2_score' in data['data1'][instruction_to_index_1[instr]]):
                future = executor.submit(call_api, q, a1, a2)
                future_to_task[future] = (instr, 'normal')
            # Reverse order evaluation (swap answer1 and answer2)
            if not ('answer_1_score_reverse' in data['data1'][instruction_to_index_1[instr]] and
                    'answer_2_score_reverse' in data['data1'][instruction_to_index_1[instr]]):
                future = executor.submit(call_api, q, a2, a1)  # Swapped a1 and a2 here
                future_to_task[future] = (instr, 'reverse')

        for future in as_completed(future_to_task):
            instr, eval_type = future_to_task[future]
            try:
                review, scores = future.result()
                idx1 = instruction_to_index_1[instr]

                # Append GPT evaluation
                if 'GPT_eval' not in data['data1'][idx1]:
                    data['data1'][idx1]['GPT_eval'] = []
                data['data1'][idx1]['GPT_eval'].append(review)

                # Write scores based on evaluation type
                if eval_type == 'normal':
                    if scores[0] != -1 and scores[1] != -1:
                        data['data1'][idx1]['answer_1_score'] = scores[0]
                        data['data1'][idx1]['answer_2_score'] = scores[1]
                elif eval_type == 'reverse':
                    if scores[0] != -1 and scores[1] != -1:
                        # In reverse order, scores[0] corresponds to answer_2_score_reverse, scores[1] to answer_1_score_reverse
                        data['data1'][idx1]['answer_2_score_reverse'] = scores[0]
                        data['data1'][idx1]['answer_1_score_reverse'] = scores[1]

            except Exception as e:
                print(f"Instruction '{instr}' with evaluation type '{eval_type}' generated an exception: {e}")

    # After all tasks complete, write back file with updated data1 only
    with open(output_file, 'w', encoding='utf-8') as out:
        json.dump(data, out, ensure_ascii=False, indent=4)

print("Processing complete.")
