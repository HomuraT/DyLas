import argparse

from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from data_utils.dataset import load_dataset
from wandb_uitls.WandbUtils import init_wandb_parser, get_wandb, wandb_log, wandb_finish, close_wandb


def main(args):
    dataset = load_dataset(**args.__dict__)

    prediction_path = args.prediction_path
    prediction = dataset.read_prediction(prediction_path)

    y = np.zeros([len(prediction), dataset.get_label_number()], int)
    y_hat = np.zeros_like(y)
    for idx, (data, pred) in tqdm(enumerate(zip(dataset, prediction))):
        assert data['ID'] == pred['ID'], 'the prediction and data are not assigned, please check the files...'
        labels = pred['labels']
        p_labels = pred['prediction']

        for i in dataset.get_indexes_of_labels(labels=labels, **args.__dict__, isLabels=True):
            y[idx, i] = 1

        for i in dataset.get_indexes_of_labels_by_mode(labels=p_labels, **args.__dict__):
            y_hat[idx, i] = 1


    # overall f1 for all labels
    print('calculating micro metric...')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y, y_hat, average='micro', zero_division=False)

    # average f1 for each label
    print('calculating macro metric...')
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y, y_hat, average='macro', zero_division=False)

    # calculate the f1 for each sample, and finally use the average of f1 as the result
    print('calculating samples metric...')
    samples_precision, samples_recall, samples_f1, _ = precision_recall_fscore_support(y, y_hat, average='samples', zero_division=False)

    # Print the metrics
    print(f"MA-p MA-r MA-f1 MI-p MI-r MI-f1 SA-p SA-r SA-f1")
    print(
        f"{macro_precision*100:.1f} {macro_recall*100:.1f} {macro_f1*100:.1f} {micro_precision*100:.1f} {micro_recall*100:.1f} {micro_f1*100:.1f} {samples_precision*100:.1f} {samples_recall*100:.1f} {samples_f1*100:.1f}")

    print()
    print(f"correct\tcorrect_only\tincorrect")
    correct_flag = (y == y_hat) & (y != 0)

    c = ((correct_flag.sum(1) > 0).sum() / len(prediction))*100
    co = (((correct_flag.sum(1) == y_hat.sum(1)) & (correct_flag.sum(1) > 0)).sum() / len(prediction))*100
    ic = ((correct_flag.sum(1) == 0).sum() / len(prediction))*100

    print(f"{c:.1f} {co:.1f} {ic:.1f}")

    print()
    print(f"avg.#pre_label")
    print(f"{(y_hat.sum() / y_hat.shape[0]):.1f}")

    wandb_log({
        'MA-p': macro_precision*100,
        'MA-r': macro_recall*100,
        'MA-f1': macro_f1*100,
        'MI-p': micro_precision*100,
        'MI-r': micro_recall*100,
        'MI-f1': micro_f1*100,
        'SA-p': samples_precision*100,
        'SA-r': samples_recall*100,
        'SA-f1': samples_f1 * 100,
        'correct': c,
        'correct only': co,
        '#pre_label': (y_hat.sum() / y_hat.shape[0]),
    })

    close_wandb()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm predict labels")
    parser.add_argument('--dataset', type=str, help='The name of the dataset. Now support [codiesp]')
    parser.add_argument('--data_path', type=str, help='The path of dataset')
    parser.add_argument('--prediction_path', type=str, help='The path of prediction')
    parser.add_argument('--mode', type=str, default='vec_sim', help='vec_sim, str_align')

    # optional
    parser.add_argument("--label_description_path", default=None, type=str, help="标签描述")
    parser.add_argument("--data_file_name", default='subset.json', type=str, help="")
    parser.add_argument("--PLM_name", default='all-MiniLM-L6-v2', type=str, help="")
    parser.add_argument("--cache_folder", default='../huggingface_models/hub', type=str, help="wn18rr预训练模型地址")
    parser.add_argument("--informations_embeddings_name", default='informations_embeddings-MiniLM.pt', type=str, help="")
    parser.add_argument("--id2information_name", default='id2information.json', type=str, help="id2information_name")
    parser.add_argument("--examples_path", default='examples.json', type=str, help="examples_path")

    parser.add_argument("--pre_mode", default='top_k', type=str, help="top_k, top_p, top_p_reverse, overall_topk")
    parser.add_argument("--pre_topk", default=1, type=int, help="在进行相似度匹配时，去前k个")
    parser.add_argument("--pre_topp", default=1.5, type=float, help="")


    ## codiesp
    parser.add_argument("--handle_mode", default='desc_dict', type=str, help="icd, desc_dict")

    ##
    init_wandb_parser(parser)

    args = parser.parse_args()

    if args.wandb:
        print('init wandb by args:', args)
        w = get_wandb(args)
        print('wandb:', w)

    print(args)
    main(args)