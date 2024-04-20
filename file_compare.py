def read_numbers_from_file(file_path):
    numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.lstrip('-').isdigit():
                numbers.append(int(line))
    return numbers

def compare_files(result_path, answer_path):
    result_numbers = read_numbers_from_file(result_path)
    answer_numbers = read_numbers_from_file(answer_path)

    # print("result:",result_numbers)
    # print("answer:",answer_numbers)

    correct_cnt = 0
    total_cnt = min(len(result_numbers), len(answer_numbers))

    for i in range(total_cnt):
        if result_numbers[i] == answer_numbers[i]:
            correct_cnt += 1

    print(f"accuracy:{correct_cnt/total_cnt:.3f}({correct_cnt}/{total_cnt})")

if __name__ == "__main__":
    compare_files('/mnt/sevenT/xyn/llama2_log/result.txt', './data/test_labels.txt')