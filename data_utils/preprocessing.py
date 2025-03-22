import json
import os
import random
import re

import pandas as pd
from tqdm import tqdm


def preprocessing_codiesp(dataset, data_path, **kwargs):
    '''
    Miranda-Escalada, Antonio et al. "Overview of Automatic Clinical Coding - Annotations, Guidelines, and Solutions for non-English Clinical Cases at CodiEsp Track of CLEF eHealth 2020.", Conference and Labs of the Evaluation Forum (2020)

    dataset download url: https://zenodo.org/records/3837305#.XsZFoXUzZpg

    label description url: https://github.com/StefanoTrv/simple_icd_10_CM/blob/master/data/icd10cm-order-Jan-2021.txt

    :param dataset:
    :param data_path:
    :return:
    '''

    assert 'final_dataset_v4_to_publish' in os.listdir(
        data_path), 'the version of the dataset may be wrong, please download the dataset in the URL https://zenodo.org/records/3837305#.XsZFoXUzZpg'

    # handle label description
    print('handle label description')
    label_description = {}
    df_label_description = pd.read_csv(os.path.join(data_path, f'icd10cm-order-Jan-2021.txt'), sep='\t', header=None)
    for index, row in tqdm(df_label_description.iterrows()):
        line = re.findall(r'([^\s]+)', row.item())
        ICD10_code = line[1]
        ICD10_desc = []
        for w in line[3:]:
            if w in ICD10_desc:
                break
            ICD10_desc.append(w)
        label_description[ICD10_code] = ' '.join(ICD10_desc)
    json.dump(label_description, open(os.path.join(data_path, f'label_description.json'), 'w', encoding='utf-8'),
              indent=2)

    dataset_base_path = os.path.join(data_path, 'final_dataset_v4_to_publish')
    data_type = ['train', 'dev', 'test']

    def handle_codiesp(dirpath):
        data_type = os.path.basename(dirpath)
        files_path = os.path.join(dirpath, 'text_files')
        files_en_path = os.path.join(dirpath, 'text_files_en')
        files = os.listdir(files_path)

        df_D = pd.read_csv(os.path.join(dirpath, f'{data_type}D.tsv'), sep='\t', header=None,
                           names=['ID', 'DIAGNOSTICO_code'])
        df_P = pd.read_csv(os.path.join(dirpath, f'{data_type}P.tsv'), sep='\t', header=None,
                           names=['Doc_ID', 'PROCEDIMIENTO_code'])
        # df_X = pd.read_csv(os.path.join(dirpath, f'{data_type}X.tsv'), sep='\t', header=None,                           names=['Doc_ID', 'Code_type', 'Code', 'Textual_evidence', 'Lvidence_location'])
        dataset = {}

        # handle text
        for fname in files:
            data = {}
            with open(os.path.join(files_path, fname), 'r', encoding='utf-8') as f:
                data['text_Spanish'] = f.readlines()
            with open(os.path.join(files_en_path, fname), 'r', encoding='utf-8') as f:
                data['text_English'] = f.readlines()
            doc_id = fname.split('.')[0]

            assert doc_id not in dataset, 'doc_id is already in the dataset. Please check the files...'
            dataset[doc_id] = data

        # handle DIAGNOSTICO_codes
        for index, row in df_D.iterrows():
            doc_id = row.iloc[0]
            code = row.iloc[1]

            # transfer to icd-10-cm
            code = code.upper()
            code = code.replace('.', '')
            if code not in label_description:
                continue

            assert doc_id in dataset, 'doc_id is not in the dataset. Please check the files...'
            if 'DIAGNOSTICO_codes' not in dataset[doc_id]:
                dataset[doc_id]['DIAGNOSTICO_codes'] = []
            dataset[doc_id]['DIAGNOSTICO_codes'].append(code)

        # handle PROCEDIMIENTO_codes
        for index, row in df_P.iterrows():
            doc_id = row.iloc[0]
            code = row.iloc[1]

            assert doc_id in dataset, 'doc_id is not in the dataset. Please check the files...'
            if 'PROCEDIMIENTO_codes' not in dataset[doc_id]:
                dataset[doc_id]['PROCEDIMIENTO_codes'] = []
            dataset[doc_id]['PROCEDIMIENTO_codes'].append(code)

        return dataset

    DIAGNOSTICO_codes = set()
    PROCEDIMIENTO_codes = set()
    for tdata in data_type:
        print(f'handling {tdata} set... ')
        dataset = handle_codiesp(os.path.join(dataset_base_path, tdata))

        for doc_id, data in dataset.items():
            if 'DIAGNOSTICO_codes' in data:
                Dcodes = data['DIAGNOSTICO_codes']
                DIAGNOSTICO_codes.update(Dcodes)
            else:
                data['DIAGNOSTICO_codes'] = []

            if 'PROCEDIMIENTO_codes' in data:
                Pcodes = data['PROCEDIMIENTO_codes']
                PROCEDIMIENTO_codes.update(Pcodes)
            else:
                data['PROCEDIMIENTO_codes'] = []

        json.dump(dataset, open(os.path.join(data_path, f'{tdata}.json'), 'w', encoding='utf-8'), indent=2)

    # handle labels
    codes = DIAGNOSTICO_codes | PROCEDIMIENTO_codes
    json.dump(list(DIAGNOSTICO_codes), open(os.path.join(data_path, f'DIAGNOSTICO_codes.json'), 'w', encoding='utf-8'),
              indent=2)
    json.dump(list(PROCEDIMIENTO_codes),
              open(os.path.join(data_path, f'PROCEDIMIENTO_codes.json'), 'w', encoding='utf-8'), indent=2)
    json.dump(list(codes), open(os.path.join(data_path, f'codes.json'), 'w', encoding='utf-8'), indent=2)


def preprocessing_WN18RR(dataset, data_path, **kwargs):
    """
    Dettmers, T., Minervini, P., Stenetorp, P., Riedel, S.: Convolutional 2d knowledge graph embeddings. In: McIlraith, S.A., Weinberger, K.Q. (eds.) Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018. pp. 1811–1818. AAAI Press (2018), https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17366

    dataset download url: https://github.com/intfloat/SimKGC/tree/main/data/WN18RR

    :param dataset:
    :param data_path:
    :return:
    """

    # read entity information
    id2information = {}
    with open(os.path.join(data_path, 'wordnet-mlj12-definitions.txt'), 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break

            msg = line.split('\t')
            id2information[msg[0]] = {
                'term': ' '.join([i for i in msg[1].split('_')[:-2] if i]).strip(),
                'words': [i for i in msg[1].split('_')[:-2] if i],
                'definition': msg[1],
                'description': msg[2].strip()
            }
    json.dump(id2information, open(os.path.join(data_path, 'id2information.json'), 'w', encoding='utf-8'), indent=3)

    data_type = ['train', 'valid', 'test']

    # build dataset
    def handle_dataset(data_path, data_type, dataset):
        fpath = os.path.join(data_path, f'{data_type}.txt')
        with open(fpath, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                head_id, relation, tail_id = line.strip().split('\t')
                key = str([head_id, relation])
                if key not in dataset:
                    dataset[key] = []

                dataset[key].append(tail_id)

    dataset = {}
    for t in data_type:
        print(f'start build {t} set...')
        handle_dataset(data_path, t, dataset)

    json.dump(dataset, open(os.path.join(data_path, 'dataset.json'), 'w', encoding='utf-8'), indent=1)

    # create example
    print('build examples...')
    relations = []
    with open(os.path.join(data_path, 'relations.dict'), 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            _, rel = line.strip().split('\t')
            relations.append(rel)

    examples = {}
    wn18rr_example_number = kwargs['example_number']
    wn18rr_example_tail_max_number = kwargs['example_tail_max_number']
    wn18rr_example_tail_min_number = kwargs['example_tail_min_number']
    count = 0
    example_number = wn18rr_example_number * len(relations)
    for hr, ts in dataset.items():
        head, rel = eval(hr)
        if rel not in examples:
            examples[rel] = []
        if len(examples[rel]) < wn18rr_example_number and wn18rr_example_tail_max_number >= len(
                ts) >= wn18rr_example_tail_min_number:
            examples[rel].append([head, ts])
            count += 1

        if count >= example_number:
            break

    json.dump(examples, open(os.path.join(data_path, 'rel2examples.json'), 'w', encoding='utf-8'), indent=1)

    # extract sub dataset
    print('extracting sub dataset...')
    sample_number = kwargs['sample_number']
    sub_dataset_name = kwargs['sub_dataset_name']

    sub_dataset = {i[0]: i[1] for i in random.sample(list(dataset.items()), k=sample_number)}
    json.dump(sub_dataset, open(os.path.join(data_path, sub_dataset_name), 'w', encoding='utf-8'), indent=1)

    print('finished...')


def preprocessing_FB15k237(dataset, data_path, **kwargs):
    """
    Toutanova, K., Chen, D.: Observed versus latent features for knowledge base and text inference. In: Proceedings of the 3rd Workshop on Continuous Vector Space Models and their Compositionality. pp. 57–66. Association for Computational Linguistics, Beijing, China (Jul 2015). https://doi.org/10.18653/v1/W15-4007

    dataset download url: https://github.com/intfloat/SimKGC/tree/main/data/FB15k237

    :param dataset:
    :param data_path:
    :param kwargs:
    :return:
    """

    # read entity information
    id2information = {}
    with open(os.path.join(data_path, 'FB15k_mid2name.txt'), 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break

            msg = line.split('\t')
            id2information[msg[0]] = {
                'name': msg[1].strip()
            }

    with open(os.path.join(data_path, 'FB15k_mid2description.txt'), 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break

            msg = line.split('\t')
            id2information[msg[0]]['description'] = msg[1].strip()

    json.dump(id2information, open(os.path.join(data_path, 'id2information.json'), 'w', encoding='utf-8'), indent=3)

    data_type = ['train', 'valid', 'test']

    relations = set()

    # build dataset
    def handle_dataset(data_path, data_type, dataset, relations):
        fpath = os.path.join(data_path, f'{data_type}.txt')
        with open(fpath, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                head_id, relation, tail_id = line.strip().split('\t')
                relations.add(relation)
                key = str([head_id, relation])
                if key not in dataset:
                    dataset[key] = []

                dataset[key].append(tail_id)

    dataset = {}
    for t in data_type:
        print(f'start build {t} set...')
        handle_dataset(data_path, t, dataset, relations)
    json.dump(dataset, open(os.path.join(data_path, 'dataset.json'), 'w', encoding='utf-8'), indent=1)
    relations = list(relations)

    # extract sub dataset
    print('extracting sub dataset...')
    sample_number = kwargs['sample_number']
    sub_dataset_name = kwargs['sub_dataset_name']

    sub_dataset = {i[0]: i[1] for i in random.sample(list(dataset.items()), k=sample_number)}
    json.dump(sub_dataset, open(os.path.join(data_path, sub_dataset_name), 'w', encoding='utf-8'), indent=1)

    # create example
    print('build examples...')

    examples = {}
    example_number = kwargs['example_number']
    example_tail_max_number = kwargs['example_tail_max_number']
    example_tail_min_number = kwargs['example_tail_min_number']
    count = 0
    total_example_number = example_number * len(relations)
    for hr, ts in dataset.items():
        head, rel = eval(hr)
        if rel not in examples:
            examples[rel] = []
        if len(examples[rel]) < example_number and example_tail_max_number >= len(
                ts) >= example_tail_min_number:
            examples[rel].append([head, ts])
            count += 1

        if count >= total_example_number:
            break

    json.dump(examples, open(os.path.join(data_path, 'rel2examples.json'), 'w', encoding='utf-8'), indent=1)

    json.dump(relations, open(os.path.join(data_path, 'relations.json'), 'w', encoding='utf-8'), indent=1)

    print('finished...')

def preprocessing_Reuters21578(dataset, data_path, **kwargs):
    """
    Padmanabhan, Divya, et al. "Topic model based multi-label classification." 2016 IEEE 28th International Conference on Tools with Artificial Intelligence (ICTAI). IEEE, 2016.

    dataset download url: https://www.kaggle.com/datasets/nltkdata/reuters/data

    :param dataset:
    :param data_path:
    :param kwargs:
    :return:
    """

    # load id and labels
    lpath = os.path.join(data_path, 'cats.txt')
    id2labels = {}
    labels = set()
    with open(lpath, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            msg = line.strip().split(' ')
            id2labels[msg[0]] = msg[1:]

            [labels.add(l) for l in msg[1:]]

    json.dump(list(labels), open(os.path.join(data_path, 'labels.json'), 'w', encoding='utf-8'), indent=1)
    json.dump({i:l for i, l in enumerate(list(labels))}, open(os.path.join(data_path, 'id2information.json'), 'w', encoding='utf-8'), indent=1)

    # handle training and testing
    def handle_dataset(path, dtype):
        dataset = {}
        dname = 'training' if dtype == 'train' else 'test'
        dpath = os.path.join(path, dname)
        fnames = os.listdir(dpath)
        for fname in fnames:
            fpath = os.path.join(dpath, fname)
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                text =' '.join([i.strip() for i in f.readlines()])
            ID = f'{dname}/{fname}'
            dataset[ID] = {
                'text': text,
                'labels': id2labels[ID],
            }
        return dataset

    dtypes = ['train', 'test']
    datasets = {}
    for dtype in dtypes:
        print(f'handle {dtype} set...')
        dataset = handle_dataset(data_path, dtype)
        datasets[dtype] = dataset
        json.dump(dataset, open(os.path.join(data_path, f'{dtype}.json'), 'w', encoding='utf-8'), indent=1)

    # extract sub dataset
    print('extracting sub dataset...')
    sample_number = kwargs['sample_number']
    sub_dataset_name = kwargs['sub_dataset_name']

    sub_dataset = {i[0]: i[1] for i in random.sample(list(datasets['test'].items()), k=sample_number)}
    json.dump(sub_dataset, open(os.path.join(data_path, sub_dataset_name), 'w', encoding='utf-8'), indent=1)

    # create example
    print('build examples...')

    example_number = kwargs['example_number']
    example_tail_min_number = kwargs['example_tail_min_number']
    examples = {}

    count = 0
    total_example_number = example_number
    for k, v in datasets['train'].items():
        rel_num = len(v['labels'])
        if count < example_number and rel_num >= example_tail_min_number:
            examples[k]=v
            count += 1

        if count >= total_example_number:
            break

    json.dump(examples, open(os.path.join(data_path, 'examples.json'), 'w', encoding='utf-8'), indent=1)


def preprocessing(**kwargs):
    dataset = kwargs['dataset']
    data_path = kwargs['data_path']
    print(f'Start to handle {dataset}')
    print(f'Data path is {data_path}')

    if dataset == 'codiesp':
        preprocessing_codiesp(**kwargs)
    elif dataset == 'WN18RR':
        preprocessing_WN18RR(**kwargs)
    elif dataset == 'FB15k237':
        preprocessing_FB15k237(**kwargs)
    elif dataset == 'Reuters21578':
        preprocessing_Reuters21578(**kwargs)
