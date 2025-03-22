import argparse

from data_utils.preprocessing import preprocessing

from myUtils.utils import set_random_seed


def main(args):
    set_random_seed(args.random_seed)
    preprocessing(**args.__dict__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data preprocessing")

    parser.add_argument('--dataset', type=str, help='The name of the dataset. Now support [codiesp, WN18RR]')
    parser.add_argument('--data_path', type=str, help='The path of dataset')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')

    # optional
    parser.add_argument('--sample_number', type=int, default=500, help='抽取数据集数量', required=False)
    parser.add_argument('--sub_dataset_name', type=str, default='subset_500.json', help='抽取数据集文件名称', required=False)
    parser.add_argument('--example_number', type=int, default=2, help='', required=False)
    parser.add_argument('--example_tail_max_number', type=int, default=5, help='', required=False)
    parser.add_argument('--example_tail_min_number', type=int, default=2, help='', required=False)

    args = parser.parse_args()

    print(args)
    main(args)