import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import json

# 初始化OpenAI客户端（请使用安全方式管理您的API密钥）
client = OpenAI(api_key="sk-Lm87MVS3hAwZjEJIvoUTMpoyTHq81ecxkZTF5MUhoVT3BlbkFJziHOsIbDS9Q8kdDWpVkUdsR_EW91KiK5QKNazdROAA")  # 请替换为您的API密钥

# 输入与输出文件夹路径
input_dir_1 = r"V:\ry\train2.0\eval\alpacagpt35_slef500_lama27"  # 第一个批次的输入文件夹
input_dir_2 = r"V:\ry\train2.0\eval\alpaca_gpt35_result\llama_alpaca_gpt35\llama2_alpaca_gpt35_52k"  # 第二个批次的输入文件夹
output_dir = r"self1.0(500)_vs_quan52k_alpacagpt35_llama27"  # 单一输出文件夹

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

# 获取两个文件夹中所有json文件的文件名（不含路径）
files_1 = {os.path.basename(f) for f in glob.glob(os.path.join(input_dir_1, "*.json"))}
files_2 = {os.path.basename(f) for f in glob.glob(os.path.join(input_dir_2, "*.json"))}

# 找出两个文件夹中同时存在的文件
common_files = files_1.intersection(files_2)

for filename in common_files:
    file_path_1 = os.path.join(input_dir_1, filename)
    file_path_2 = os.path.join(input_dir_2, filename)

    output_file = os.path.join(output_dir, filename)

    # 如果输出文件已存在则从输出文件加载，否则从输入文件加载
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as pf:
            data = json.load(pf)
    else:
        with open(file_path_1, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        with open(file_path_2, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
        
        # 将 data2 的 '模型输出' 添加到 data1 中作为 '模型输出2'
        for item1, item2 in zip(data1, data2):
            item1['模型输出2'] = item2.get('模型输出', '')
        
        data = {
            "data1": data1
        }

    # 建立 data1 的索引
    data1_by_instruction = {item['instruction']: item for item in data['data1']}

    # 获取 data2 的 '模型输出2'
    data1_instructions = set(item['instruction'] for item in data['data1'])

    # 找出需要处理的条目（缺少分数的指令）
    tasks = []
    instruction_to_index_1 = {e['instruction']: i for i, e in enumerate(data['data1'])}

    for entry1 in data['data1']:
        instruction = entry1.get('instruction', '')
        if instruction in data1_instructions:
            has_score_normal = ('answer_1_score' in entry1) and ('answer_2_score' in entry1)
            has_score_reverse = ('answer_1_score_reverse' in entry1) and ('answer_2_score_reverse' in entry1)

            # 如果还没有分数，才需要处理
            if not has_score_normal or not has_score_reverse:
                input_text = entry1.get('input', '')
                question = f"{instruction} Input: {input_text}"
                answer_1 = entry1.get('模型输出', '')
                answer_2 = entry1.get('模型输出2', '')
                tasks.append((instruction, question, answer_1, answer_2))

    if not tasks:
        # 没有需要处理的条目，直接写回（确保数据最新）
        with open(output_file, 'w', encoding='utf-8') as out:
            json.dump(data, out, ensure_ascii=False, indent=4)
        continue

    # 并行处理任务
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_task = {}
        for (instr, q, a1, a2) in tasks:
            # 正常顺序评估
            if not ('answer_1_score' in data['data1'][instruction_to_index_1[instr]] and
                    'answer_2_score' in data['data1'][instruction_to_index_1[instr]]):
                future = executor.submit(call_api, q, a1, a2)
                future_to_task[future] = (instr, 'normal')
            # 反序评估（交换 answer1 和 answer2）
            if not ('answer_1_score_reverse' in data['data1'][instruction_to_index_1[instr]] and
                    'answer_2_score_reverse' in data['data1'][instruction_to_index_1[instr]]):
                future = executor.submit(call_api, q, a2, a1)  # 这里交换了 a1 和 a2
                future_to_task[future] = (instr, 'reverse')

        for future in as_completed(future_to_task):
            instr, eval_type = future_to_task[future]
            try:
                review, scores = future.result()
                idx1 = instruction_to_index_1[instr]

                # 写入GPT_eval
                if 'GPT_eval' not in data['data1'][idx1]:
                    data['data1'][idx1]['GPT_eval'] = []
                data['data1'][idx1]['GPT_eval'].append(review)

                # 根据评估类型写入分数
                if eval_type == 'normal':
                    if scores[0] != -1 and scores[1] != -1:
                        data['data1'][idx1]['answer_1_score'] = scores[0]
                        data['data1'][idx1]['answer_2_score'] = scores[1]
                elif eval_type == 'reverse':
                    if scores[0] != -1 and scores[1] != -1:
                        # 反序时，scores[0] 对应 answer_2_score_reverse，scores[1] 对应 answer_1_score_reverse
                        data['data1'][idx1]['answer_2_score_reverse'] = scores[0]
                        data['data1'][idx1]['answer_1_score_reverse'] = scores[1]

            except Exception as e:
                print(f"Instruction '{instr}' with evaluation type '{eval_type}' generated an exception: {e}")

    # 所有任务完成后统一写回文件，只写入 data1
    with open(output_file, 'w', encoding='utf-8') as out:
        json.dump(data, out, ensure_ascii=False, indent=4)

print("Processing complete.")