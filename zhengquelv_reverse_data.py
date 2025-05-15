import os
import json

def create_entry_key(entry):
    """
    Creates a unique key for an entry based on 'instruction' and other identifiers if they exist.

    优先使用 'id', 'question_id', 或 'idx' 作为额外的标识符来创建唯一键。

    参数:
        entry (dict): JSON 文件中的一个条目。

    返回:
        str: 唯一的键字符串。
    """
    key_parts = [entry.get('instruction', '')]
    for identifier in ['id', 'question_id', 'idx']:
        if identifier in entry:
            key_parts.append(str(entry[identifier]))
    return '|'.join(key_parts)

def get_scores_all(data1, data2):
    """
    计算两个数据列表中的胜、平、负情况。

    参数:
        data1 (list): 第一个 JSON 文件的数据列表。
        data2 (list): 第二个 JSON 文件的数据列表。

    返回:
        list: 包含 win_count, tie_count, lose_count 的列表。
    """
    score1, score2, score3 = 0, 0, 0

    # 创建 data2 的映射，以便快速查找匹配的条目
    data2_map = {}
    for entry in data2:
        key = create_entry_key(entry)
        data2_map[key] = entry

    for entry1 in data1:
        key = create_entry_key(entry1)
        entry2 = data2_map.get(key)

        if not entry2:
            print(f"未找到匹配的条目，键: {key}")
            continue  # 或者根据需要处理未匹配的条目

        # 提取分数
        k1_score = entry1.get('answer_1_score', 0)
        k2_score = entry1.get('answer_2_score', 0)

        k1_score_reverse = entry2.get('answer_1_score_reverse', 0)
        k2_score_reverse = entry2.get('answer_2_score_reverse', 0)

        # 按照给定规则计算胜负情况
        if k1_score > k2_score and k1_score_reverse > k2_score_reverse:
            score1 += 1
        elif k1_score < k2_score and k1_score_reverse > k2_score_reverse:
            score2 += 1
        elif k1_score > k2_score and k1_score_reverse < k2_score_reverse:
            score2 += 1
        elif k1_score == k2_score and k1_score_reverse > k2_score_reverse:
            score1 += 1
        elif k1_score > k2_score and k1_score_reverse == k2_score_reverse:
            score1 += 1
        elif k1_score == k2_score and k1_score_reverse < k2_score_reverse:
            score3 += 1
        elif k1_score < k2_score and k1_score_reverse == k2_score_reverse:
            score3 += 1
        elif k1_score == k2_score and k1_score_reverse == k2_score_reverse:
            score2 += 1
        elif k1_score < k2_score and k1_score_reverse < k2_score_reverse:
            score3 += 1

    return [score1, score2, score3]

def analyze_json_files(file1_path, file2_path):
    """
    分析两个 JSON 文件，计算胜、平、负的条目数量，并返回结果。

    参数:
        file1_path (str): 第一个 JSON 文件的路径。
        file2_path (str): 第二个 JSON 文件的路径。

    返回:
        dict: 包含 win_count, tie_count, lose_count, accuracy 和 total 的字典。
    """
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
    except json.JSONDecodeError as e:
        print(f"无法解析文件 {file1_path} 或 {file2_path}: {e}")
        return None
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        return None

    # 提取两组数据，根据 JSON 结构判断
    if isinstance(data1, dict):
        pure_data1 = data1.get('data1', [])
        if not isinstance(pure_data1, list):
            print(f"文件 {file1_path} 中的 'data1' 不是一个列表。")
            pure_data1 = []
    elif isinstance(data1, list):
        pure_data1 = data1
    else:
        print(f"文件 {file1_path} 的 JSON 结构不符合预期。")
        pure_data1 = []

    if isinstance(data2, dict):
        pure_data2 = data2.get('data1', [])
        if not isinstance(pure_data2, list):
            print(f"文件 {file2_path} 中的 'data1' 不是一个列表。")
            pure_data2 = []
    elif isinstance(data2, list):
        pure_data2 = data2
    else:
        print(f"文件 {file2_path} 的 JSON 结构不符合预期。")
        pure_data2 = []

    scores = get_scores_all(pure_data1, pure_data2)
    total_matched = scores[0] + scores[1] + scores[2]
    return {
        'win_count': scores[0],
        'tie_count': scores[1],
        'lose_count': scores[2],
        'accuracy': (scores[0] / total_matched) * 100 if total_matched > 0 else 0,
        'total': total_matched
    }

def traverse_directories(folder1_path, folder2_path):
    """
    遍历两个指定的文件夹，分析每对同名的 JSON 文件。

    参数:
        folder1_path (str): 第一个文件夹的路径。
        folder2_path (str): 第二个文件夹的路径。
    """
    if not os.path.isdir(folder1_path):
        print(f"第一个文件夹路径无效: {folder1_path}")
        return

    if not os.path.isdir(folder2_path):
        print(f"第二个文件夹路径无效: {folder2_path}")
        return

    # 获取两个文件夹中的所有 JSON 文件
    json_files1 = set(f for f in os.listdir(folder1_path) if f.endswith('.json'))
    json_files2 = set(f for f in os.listdir(folder2_path) if f.endswith('.json'))

    # 找到两个文件夹中共有的 JSON 文件
    common_files = sorted(json_files1.intersection(json_files2))

    if not common_files:
        print("在两个文件夹中未找到同名的 JSON 文件。")
        return

    for filename in common_files:
        file1_path = os.path.join(folder1_path, filename)
        file2_path = os.path.join(folder2_path, filename)
        result = analyze_json_files(file1_path, file2_path)
        if result is not None:
            print(f"文件: {filename}")
            print(f"  answer_1_score > answer_2_score 的条目数 (win_count): {result['win_count']}")
            print(f"  answer_1_score == answer_2_score 的条目数 (tie_count): {result['tie_count']}")
            print(f"  answer_1_score < answer_2_score 的条目数 (lose_count): {result['lose_count']}")
            print(f"  正确率 (win_count / total): {result['accuracy']:.2f}%")
            print(f"  匹配的总条目数: {result['total']}")
            print("\n")

if __name__ == "__main__":
    # 设置要遍历的两个文件夹路径
    folder1_path = r"V:\ry\train2.0\eval\llama_alpacagpt4\self vs quan52K_zf_alpacagpt4_llama2\self_1000_test_dir_quan"  # 请根据实际情况更改路径
    folder2_path = r"V:\ry\train2.0\eval\llama_alpacagpt4\self vs quan52K_zf_alpacagpt4_llama2\self1.0(gpt4) vs Alpaca_52K(gpt4)_llama2_reverse"  # 请根据实际情况更改路径
    traverse_directories(folder1_path, folder2_path)