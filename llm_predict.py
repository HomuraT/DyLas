import argparse
import os

from tqdm import tqdm

from data_utils.dataset import load_dataset
from llm.utils.gpt_utils import OpenAI_API, API_Manager, OpenAI_config
from llm.utils.prompt_utils import load_prompt, Prompt_hepler
from myUtils.utils import set_random_seed


def main(args):
    set_random_seed(args.random_seed)

    dataset = args.dataset
    data_path = args.data_path
    prediction_path = args.prediction_path
    prompt_path = args.prompt_path
    prompt_id = args.prompt_id
    api_name = args.api_name
    max_num_this_run = args.max_num_this_run
    error_extraction_count = args.error_extraction_count

    dataset = load_dataset(**args.__dict__)

    api: OpenAI_API = API_Manager('llm/resources/ampi.json').load().get_api_by_id(args.api_name)
    print('载入api: ', api_name, api.url, api.model, '\n')

    prompt = load_prompt(prompt_path, prompt_id)
    ph = Prompt_hepler()
    print('使用提示：', prompt_id, '\n', prompt['content'], '\n')

    print('开始运行...')
    if not os.path.exists(prediction_path):
        if not os.path.exists(os.path.dirname(prediction_path)):
            os.makedirs(os.path.dirname(prediction_path))
        prediction = []
    else:
        prediction = dataset.read_prediction(prediction_path)
    prediction_IO = open(prediction_path, 'a', encoding='utf-8')

    flag_start = len(prediction)
    num_this_run = max_num_this_run
    if num_this_run < 0:
        flag_end = len(dataset)
    else:
        flag_end = flag_start + num_this_run

    print('本次运行个数为:', flag_end-flag_start, '\n')

    for data in tqdm(dataset[flag_start:flag_end]):
        handled_result = None
        llm_result = None
        for _ in range(error_extraction_count):
            msg = ph.replace_with_dict(prompt['content'], data, '{', '}')
            try:
                llm_result = api.chat_without_history(msg)
            except Exception as e:
                print('error:', e)
                continue
            # llm_result = api.chat_without_history(msg)

            handled_result = dataset.handle_result4LLM(llm_result)
            if handled_result:
                break
        p = dataset.save_one_prediction(current_data=data, IO=prediction_IO, prediction=handled_result, LLM_response=llm_result)
        prediction.append(p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm predict labels")
    parser.add_argument('--dataset', type=str, help='The name of the dataset. Now support [codiesp]')
    parser.add_argument('--data_path', type=str, help='The path of dataset')
    parser.add_argument("--prediction_path", type=str, help="存放预测结果的路径", required=True)
    parser.add_argument("--prompt_path", type=str, help="存放提示的文件路径，提示是json文件", required=True)
    parser.add_argument("--prompt_id", type=str, help="提示的id", required=True)
    parser.add_argument("--api_name", type=str, help="调用api的名字", required=True)
    parser.add_argument("--max_num_this_run", type=int, default=-1, help="本次运行跑几个样例，-1为所有")
    parser.add_argument("--error_extraction_count", type=int, default=3, help="错误抽取尝试")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument("--informations_embeddings_name", default='informations_embeddings-MiniLM.pt', type=str, help="")
    parser.add_argument("--PLM_name", default='all-MiniLM-L6-v2', type=str, help="")

    # optional
    parser.add_argument("--cache_folder", default=None, type=str, help="")
    parser.add_argument("--data_file_name", default='subset.json', type=str, help="name of data file")
    parser.add_argument("--temperature", default=0, type=float, help="llm temperature")
    parser.add_argument("--use_chroma", default=False, type=bool, help="")
    parser.add_argument("--rag_example_num", default=1, type=int, help="")
    parser.add_argument("--examples_path", default='examples.json', type=str, help="examples_path")
    parser.add_argument("--id2information_name", default='id2information.json', type=str, help="id2information_name")



    ## codiesp
    parser.add_argument("--label_description_path", default=None, type=str, help="标签描述")
    parser.add_argument("--handle_mode", default='desc_dict', type=str, help="icd, desc_dict")

    ## WN18RR
    parser.add_argument("--max_pre_number4LLM", default=5, type=int, help="建议大模型不要预测超过的数量")



    args = parser.parse_args()
    OpenAI_config['temperature'] = args.temperature
    print(args)
    main(args)