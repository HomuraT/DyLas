import argparse
import json
import os.path

from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from data_utils.dataset import load_dataset
from llm.utils.prompt_utils import load_prompt, Prompt_hepler


def main(args):
    dataset = load_dataset(**args.__dict__)
    save_file_name = args.save_file_name
    data_path = args.data_path
    prompt_path = args.prompt_path
    prompt = load_prompt(prompt_path, args.prompt_id)
    ph = Prompt_hepler()

    id2filtered_label_ids = {}
    for d in tqdm(dataset):
        ID = d['ID']
        input_str = ph.replace_with_dict(prompt['content'], d, '{', '}')
        filtered_label_idxs = dataset.get_indexes_of_labels([input_str],pre_topk=args.pre_topk)
        labels = dataset.get_labels_of_indexes(filtered_label_idxs)
        id2filtered_label_ids[ID] = labels

    spath = os.path.join(data_path, save_file_name)
    json.dump(id2filtered_label_ids, open(spath, 'w', encoding='utf-8'))
    print('save filtered labels to ', spath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="filter labels")
    parser.add_argument('--dataset', type=str, help='The name of the dataset. Now support [codiesp]', required=True)
    parser.add_argument('--data_path', type=str, help='The path of dataset', required=True)
    parser.add_argument('--prompt_path', type=str, help='', required=True)
    parser.add_argument('--prompt_id', type=str, help='', required=True)
    parser.add_argument('--save_file_name', type=str, help='', required=True)
    parser.add_argument("--pre_topk", type=int, help="在进行相似度匹配时，去前k个", required=True)

    # optional
    parser.add_argument("--informations_embeddings_name", default='informations_embeddings-MiniLM.pt', type=str, help="标签描述")
    parser.add_argument("--label_description_path", default=None, type=str, help="标签描述")
    parser.add_argument("--data_file_name", default='subset.json', type=str, help="")
    parser.add_argument("--cache_folder", default='../huggingface_models/hub', type=str, help="wn18rr预训练模型地址")

    args = parser.parse_args()
    print(args)
    main(args)
