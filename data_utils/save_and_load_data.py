def read_prediction(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        line = file.readline()
        while line:
            # 使用制表符('\t')分割每一行，并去除换行符
            parts = line.strip().split('|', 1)
            # 假设每行都包含两个变量，将它们作为元组添加到数据列表中
            variable1, variable2 = parts
            data.append((variable1, variable2.split('|')))
            line = file.readline()
    return data