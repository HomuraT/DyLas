import json
import os.path
import random
import re
import os

from abc import ABC, abstractmethod

import torch
from sentence_transformers import SentenceTransformer
import chromadb

prediction_modes = ['str_align', 'vec_sim']

class MultiLabelDataset(ABC):
    def __init__(self, name, data_path):
        self.attr2id_dict = None
        self.id2idxes = None
        self.cache_folder = None
        self.PLM_name = None
        self.informations_embeddings = None
        self.device = None
        self.embedding_model = None
        self.name = name
        self.data_path = data_path

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def get_label_information(self, **kwargs):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def save_one_prediction(self, IO, current_data, prediction, LLM_response):
        pass

    @abstractmethod
    def handle_result4LLM(self, LLM_response):
        """

        :param LLM_response:
        :return: if the LLM_response is not suitable, return None. Otherwise, return a dictionary
        """
        pass

    def _load_embedding_model(self):
        if self.embedding_model is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_model = SentenceTransformer(self.PLM_name, cache_folder=self.cache_folder).to(
                self.device)
            self.informations_embeddings['informations_embeddings'] = self.informations_embeddings[
                'informations_embeddings'].to(self.device)
    def get_indexes_of_labels(self, labels, pre_topk=1, isLabels=False, **kwargs):
        self._load_embedding_model()

        idxs = []
        if not labels:
            # null
            return idxs

        if isLabels:
            for l in labels:
                if l not in idxs and l in self.id2idxes:
                    idxs.append(self.id2idxes[l])
        else:
            labels_embedding = torch.stack(
                [self.embedding_model.encode(i, convert_to_tensor=True).to(self.device) for i in labels])
            s = labels_embedding @ self.informations_embeddings['informations_embeddings'].T
            eidxs = s.topk(pre_topk).indices.flatten().tolist()
            for eidx in eidxs:
                eid = self.informations_embeddings['idx2id'][eidx]
                if eid not in idxs:
                    idxs.append(self.id2idxes[eid])

        return idxs

    @abstractmethod
    def get_labels_of_indexes(self, indexes, **kwargs):
        pass

    @abstractmethod
    def get_label_number(self):
        pass

    def read_prediction(self, prediction_path):
        prediction = []
        with open(prediction_path, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                prediction.append(json.loads(line))

        return prediction

    @abstractmethod
    def id2excepted_format(self, **kwargs):
        pass

    def get_indexes_of_labels_by_mode(self, mode='vec_sim', **kwargs):
        if mode == 'all':
            r = []
            for m in prediction_modes:
                r += self.get_indexes_of_labels_by_mode(m, **kwargs)
            return r


        if mode == 'vec_sim':
            return self.get_indexes_of_labels(**kwargs)
        else:
            return self.get_indexes_of_labels_by_mode_completion(mode=mode, **kwargs)

    def get_indexes_of_labels_by_mode_completion(self, labels, pre_topk=1, isLabels=False, mode='vec_sim', **kwargs):
        if labels is None:
            return []

        if mode == 'str_align':
            return self._str_align(labels, pre_topk=pre_topk, isLabels=False, mode='vec_sim', **kwargs)

        return []

    def _get_indexes_by_mode(self, score, pre_mode, pre_topk=1, pre_topp=1, **kwargs):
        if pre_mode == 'top_k':
            return score.topk(pre_topk).indices.flatten().tolist()
        elif pre_mode in ['top_p', 'top_p_reverse']:
            values, indices  = score.topk(10)
            idxs = []
            for inds, vals in zip(indices, values):
                p = 0
                for ind, val in zip(inds, vals):
                    p += val if pre_mode == 'top_p' else 1 / val

                    if p < pre_topp:
                        idxs.append(ind.item())
                    else:
                        break

            return idxs
        elif pre_mode == 'overall_topk':
            values, indices = score.topk(pre_topk)
            values = values.flatten()
            indices = indices.flatten()

            vv, vind = values.topk(values.__len__())

            idxs = indices[vind[:pre_topk]].tolist()
            if len(set(idxs)) < pre_topk:
                new_idxs = []
                for idx in idxs:
                    if idx not in new_idxs:
                        new_idxs.append(idx)

                for i in vind.tolist()[pre_topk:]:
                    if i not in new_idxs and len(new_idxs) < pre_topk:
                        new_idxs.append(i)
                    else:
                        break
                idxs = new_idxs
            return idxs

    def _str_align(self, labels, pre_topk=1, isLabels=False, mode='vec_sim', **kwargs):
        indexes = []

        for label in labels:
            for key in self.attr2id_dict.keys():
                match_results = re.findall(f'"{key}":.*?"(.*?)".*?,', label)
                if len(match_results) > 0:
                    value = match_results[0]
                else:
                    value = None
                if value in self.attr2id_dict[key]:
                    vid = self.attr2id_dict[key][value]
                    idx = self.id2idxes[vid]
                    if idx not in indexes:
                        indexes.append(idx)
        return indexes


class CodiespDataset(MultiLabelDataset):
    def get_indexes_of_labels(self, labels, isLabels=False, **kwargs):
        if self.embedding_model is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_model = SentenceTransformer(self.PLM_name, cache_folder=self.cache_folder).to(
                self.device)
            self.informations_embeddings['informations_embeddings'] = self.informations_embeddings[
                'informations_embeddings'].to(self.device)

        idxs = []
        if not labels:
            # null1,
            return idxs

        if labels[0] in self.id2idxes and isLabels:
            for l in labels:
                if l not in idxs:
                    idxs.append(self.id2idxes[l])
        else:
            labels_embedding = torch.stack(
                [self.embedding_model.encode(i.replace('.', ''), convert_to_tensor=True).to(self.device) for i in labels])
            s = labels_embedding @ self.informations_embeddings['informations_embeddings'].T
            eidxs = self._get_indexes_by_mode(score=s, **kwargs)
            for eidx in eidxs:
                eid = self.informations_embeddings['idx2id'][eidx]
                if eid not in idxs:
                    idxs.append(self.id2idxes[eid])

        return idxs

    def id2excepted_format(self, IDs, **kwargs):
        IDs = list(set(IDs))
        if self.handle_mode in ['icd', 'icd2desc_dict']:
            msg = ''
            for ID in IDs:
                desc = self.id2label_description[ID]
                msg += f'{desc}: <code>{ID}</code>\n'
            msg += '<summary>' + '; '.join([ID for ID in IDs]) + '</summary>'
            return msg
        else:
            msg = ''
            for ID in IDs:
                desc = self.id2label_description[ID]
                msg += '<icd>{"code": "' + ID + '": "'+ desc+'"}</icd>\n'
            return msg

    def get_labels_of_indexes(self, indexes, **kwargs):
        pass

    def __init__(self, name, data_path, data_file_name, label_description_path=None, cache_folder=None,
                 id2information_name='id2information.json',
                 informations_embeddings_name='informations_embeddings-MiniLM.pt',
                 PLM_name='all-MiniLM-L6-v2',
                 handle_mode='icd',
                 use_chroma=False,
                 rag_example_num=1,
                 **kwargs):
        super().__init__(name, data_path)
        self.data_file_name = data_file_name
        self.cache_folder = cache_folder
        self.dataset_list = None
        self.id2information = None
        self.id2idxes = None
        self.informations_embeddings = None
        self.id2label_description = None
        self.PLM_name = PLM_name
        self.informations_embeddings_name = informations_embeddings_name
        self.id2information_name = id2information_name
        self.dataset = None
        self.label_description = None
        self.label2idx = None
        self.label_description_path = label_description_path
        self.handle_mode = handle_mode
        self.rag_example_num = rag_example_num

        self.chroma_name = None
        self.chroma_collection = None
        if use_chroma:
            self.chroma_clinet = chromadb.PersistentClient(path='./chroma')
            self.chroma_name = f'codiesp_resources_{os.path.basename(PLM_name)}'
            self.chroma_collection = self.chroma_clinet.get_collection(self.chroma_name)

        self.load_dataset()

    def handle_data2dict(self, data):
        self._load_embedding_model()

        if self.chroma_collection:
            en_text = data['text_En']
            k_examples = self.chroma_collection.query(self.embedding_model.encode([en_text]), n_results=self.rag_example_num)

            rag_example = ''
            rag_example_only_labels = ''
            for exdata in k_examples['metadatas'][0]:
                elabels = exdata['labels']
                etext_en = exdata['text_English']
                rag_example += '\nExample text:\n' + etext_en + '\n Example Response:\n'
                rag_example_only_labels += '\nExample Response:\n'

                ids = elabels.split(';')[:3]
                rag_example += self.id2excepted_format(ids) + '\n'
                rag_example_only_labels += self.id2excepted_format(ids) + '\n'

            data['rag_example'] = rag_example
            data['rag_example_only_labels'] = rag_example_only_labels
        return data
    def __getitem__(self, item):
        if type(item) is int:
            return self.handle_data2dict(self.dataset_list[item])
        else:
            return [self[i] for i in range(len(self))[item]]

    def __len__(self):
        return len(self.dataset)

    def get_label_information(self, label):
        pass

    def load_dataset(self):
        self.dataset = json.load(open(os.path.join(self.data_path, self.data_file_name), 'r', encoding='utf-8'))
        self.dataset_list = [
            {'ID': i[0],
             'text': ''.join(i[1]['text_English']) + '\n' + ''.join(i[1]['text_Spanish']),
             'text_En': ''.join(i[1]['text_English']),
             'text_Sp': ''.join(i[1]['text_Spanish']),
             **i[1]} for i in
            list(self.dataset.items())]
        self.informations_embeddings = torch.load(
            os.path.join(self.data_path, self.informations_embeddings_name))


        if self.label_description_path:
            self.id2label_description = json.load(
                open(os.path.join(self.data_path, self.label_description_path), 'r', encoding='utf-8'))
            self.id2idxes = {key: index for index, (key, value) in enumerate(self.id2label_description.items())}

            self.attr2id_dict = {'code':{}, 'description':{}}
            for k, v in self.id2label_description.items():
                self.attr2id_dict['code'][k] = k
                self.attr2id_dict['description'][v] = k

    def save_one_prediction(self, IO, current_data, prediction, LLM_response):
        saved_data = {
            'ID': current_data['ID'],
            'LLM_response': LLM_response,
            'prediction': prediction,
            'labels': current_data['DIAGNOSTICO_codes']
        }

        IO.write(json.dumps(saved_data) + '\n')
        IO.flush()
        return saved_data

    def handle_result4LLM(self, LLM_response):
        if self.handle_mode in ['f3_icd', 'icd']:
            icd_codes = set()
            summary_result = re.findall(r'<summary>(.*)</summary>', LLM_response)
            for r in summary_result:
                for c in r.split(';'):
                    c = c.strip().upper().replace('.', '')
                    if self.handle_mode == 'f3_icd':
                        c = c[:3]
                    if c in self.id2label_description:
                        icd_codes.add(c)

            code_result = re.findall(r'<code>(.*?)</code>', LLM_response)
            for r in code_result:
                for c in r.split(';'):
                    c = c.strip().upper().replace('.', '')
                    if self.handle_mode == 'f3_icd':
                        c = c[:3]
                    if c in self.id2label_description:
                        icd_codes.add(c)

            if len(icd_codes) == 0:
                for c in LLM_response.split(';'):
                    c = c.strip().upper().replace('.', '')
                    if self.handle_mode == 'f3_icd':
                        c = c[:3]
                    if c in self.id2label_description:
                        icd_codes.add(c)

            if len(icd_codes) == 0:
                return None
            else:
                return list(icd_codes)
        elif self.handle_mode in ['desc_dict', 'desc_dict2']:
            summary_result = re.findall(r'<icd>(.*)</icd>', LLM_response)
            return summary_result
        elif self.handle_mode in ['icd2desc_dict']:
            results = []
            for desc, code in re.findall(r'(.*?)<code>(.*?)</code>', LLM_response):
                desc = desc.strip()
                code = code.strip()
                if ':' in desc:
                    desc = desc[:-1]
                results.append(json.dumps({'code': code,'description': desc}))

            return results


    # def get_indexes_of_labels(self, labels: list, **kwargs):
    #     if not labels:
    #         return []
    #     indexes = []
    #     for l in labels:
    #         assert l in self.id2idxes, f'Label {l} is not in label set.'
    #         indexes.append(self.id2idxes[l])
    #     return indexes

    def get_label_number(self):
        return len(self.id2label_description)

    def _str_align(self, labels, pre_topk=1, isLabels=False, mode='vec_sim', **kwargs):
        if self.handle_mode == 'icd':
            indexes = []

            for label in labels:
                value = label
                if value in self.attr2id_dict['code']:
                    vid = self.attr2id_dict['code'][value]
                    idx = self.id2idxes[vid]
                    if idx not in indexes:
                        indexes.append(idx)
            return indexes
        elif self.handle_mode in ['desc_dict', 'icd2desc_dict']:
            indexes = []

            for label in labels:
                for key in self.attr2id_dict.keys():
                    match_results = re.findall(f'"{key}":.*?"(.*?)".*?,', label)
                    if len(match_results) > 0:
                        value = match_results[0]
                    else:
                        value = None
                    if key == 'code' and value is not None:
                        value = value.replace('.', '').upper()
                    if value in self.attr2id_dict[key]:
                        vid = self.attr2id_dict[key][value]
                        idx = self.id2idxes[vid]
                        if idx not in indexes:
                            indexes.append(idx)
            return indexes
        elif self.handle_mode == 'desc_dict2':
            indexes = []

            for label in labels:
                if ':' in label:
                    code, desc = label.split(':', maxsplit=1)
                else:
                    continue
                d = {'code': code.strip(), 'description': desc.strip()}
                for key in self.attr2id_dict.keys():
                    value = d[key]
                    if key == 'code' and value is not None:
                        value = value.replace('.', '').upper()
                    if value in self.attr2id_dict[key]:
                        vid = self.attr2id_dict[key][value]
                        idx = self.id2idxes[vid]
                        if idx not in indexes:
                            indexes.append(idx)
            return indexes


class WN18RRDataset(MultiLabelDataset):
    def get_labels_of_indexes(self, indexes, **kwargs):
        return [self.informations_embeddings['idx2id'][i] for i in indexes]

    def __init__(self, name, data_path, data_file_name, cache_folder=None,
                 label_description_path=None,
                 max_pre_number4LLM=10,
                 id2information_name='id2information.json',
                 informations_embeddings_name='informations_embeddings-MiniLM.pt',
                 PLM_name='all-MiniLM-L6-v2',
                 **kwargs):
        super().__init__(name, data_path)
        self.data_file_name = data_file_name
        self.label_description_path = label_description_path
        self.cache_folder = cache_folder
        self.embedding_model = None
        self.rel2examples = None
        self.id2idxes = None
        self.informations_embeddings = None
        self.sample_idxes = None
        self.id2information = None
        self.dataset = None
        self.id2label_description = None
        self.max_pre_number4LLM = max_pre_number4LLM
        self.id2information_name = id2information_name
        self.informations_embeddings_name = informations_embeddings_name
        self.PLM_name = PLM_name
        self.load_dataset()

    def get_label_information(self, ID):
        '''
        存储类别描述，可能是top50之类的东西

        :param ID:
        :return:
        '''
        if self.id2label_description is None:
            return None
        labels = self.id2label_description[ID]
        label_information = ''

        informations = []
        for l in labels:
            try:
                informations.append(self.id2information[l]['term'])
            except KeyError as e:
                return ''

        label_information += '; '.join(informations)
        return label_information

    def handle_data2dict(self, data):
        hr, ts = data
        head, relation = eval(hr)

        head_information = self.id2information[head]

        ky = {
            'relation': ' '.join(relation.split('_')).strip(),
            **head_information,
            'labels': ts,
            'examples': self.build_example_text(relation),
            'ID': hr,
            'filtered_labels': self.get_label_information(hr),
            'max_pre_number4LLM': self.max_pre_number4LLM,
        }

        return ky

    def build_example_text(self, rel):
        examples = self.rel2examples[rel]
        example_text = ''
        try:
            for h, ts in examples:
                example_text += 'Subject:\n'
                h_information = self.id2information[h]
                example_text += f'\tterm: {" ".join(h_information["words"])}\n'
                example_text += f'\tdefinition: {h_information["definition"]}\n'
                example_text += f'\tdescription: {h_information["description"]}\n\n'

                example_text += f'Predicate: {rel.replace("_", " ").strip()}\n\n'

                example_text += 'Objects:\n'
                for t in ts:
                    t_information = self.id2information[t]
                    t_kv = {**t_information}
                    del t_kv['words']
                    example_text += f"<Object>{json.dumps(t_kv)}</Object>\n"
                example_text += '\n'
        except KeyError as e:
            return ''

        return example_text

    def __getitem__(self, item):
        if type(item) is int:
            return self.handle_data2dict(self.dataset[item])
        else:
            return [self[i] for i in range(len(self))[item]]

    def __len__(self):
        return len(self.dataset)

    def save_one_prediction(self, IO, current_data, prediction, LLM_response):
        saved_data = {
            'ID': current_data['ID'],
            'LLM_response': LLM_response,
            'prediction': prediction,
            'labels': current_data['labels']
        }
        IO.write(json.dumps(saved_data) + '\n')
        IO.flush()
        return saved_data

    def handle_result4LLM(self, LLM_response):
        objs = re.findall(r'<Object>(.*?)</Object>', LLM_response)
        objs = list(set(objs))
        objs.append(LLM_response)
        return objs

    def get_indexes_of_labels(self, labels, pre_topk=1, **kwargs):
        if self.embedding_model is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_model = SentenceTransformer(self.PLM_name, cache_folder=self.cache_folder).to(
                self.device)
            self.informations_embeddings['informations_embeddings'] = self.informations_embeddings[
                'informations_embeddings'].to(self.device)

        idxs = []
        if not labels:
            # null
            return idxs

        if labels[0] in self.id2idxes:
            for l in labels:
                if l not in idxs:
                    idxs.append(self.id2idxes[l])
        else:
            labels_embedding = torch.stack(
                [self.embedding_model.encode(i, convert_to_tensor=True).to(self.device) for i in labels])
            s = labels_embedding @ self.informations_embeddings['informations_embeddings'].T
            eidxs = s.topk(pre_topk).indices.flatten().tolist()
            for eidx in eidxs:
                eid = self.informations_embeddings['idx2id'][eidx]
                if eid not in idxs:
                    idxs.append(self.id2idxes[eid])

        return idxs

    def get_label_number(self):
        return len(self.id2information)

    def load_dataset(self):
        self.dataset = json.load(open(os.path.join(self.data_path, self.data_file_name), 'r', encoding='utf-8'))
        self.rel2examples = json.load(open(os.path.join(self.data_path, 'rel2examples.json'), 'r', encoding='utf-8'))
        self.dataset = list(self.dataset.items())
        self.id2information = json.load(
            open(os.path.join(self.data_path, self.id2information_name), 'r', encoding='utf-8'))
        self.id2idxes = {k: i for i, (k, v) in enumerate(list(self.id2information.items()))}
        self.informations_embeddings = torch.load(os.path.join(self.data_path, self.informations_embeddings_name))

        if self.label_description_path:
            self.id2label_description = json.load(
                open(os.path.join(self.data_path, self.label_description_path), 'r', encoding='utf-8'))

        # 对齐
        self.attr2id_dict = {}
        for k, v in self.id2information.items():
            for vk, vv in v.items():
                if vk not in self.attr2id_dict:
                    self.attr2id_dict[vk] = {}
                self.attr2id_dict[vk][str(vv)] = k  # to id

    def id2excepted_format(self, IDs, **kwargs):
        ifs = []
        for ID in IDs:
            information = {**self.id2information[ID]}
            del information['words']
            ifstr = f'<Object>{json.dumps(information)}</Object>'
            if ifstr not in ifs:
                ifs.append(ifstr)

        return '\n'.join(ifs)


class FB15k237Dataset(MultiLabelDataset):
    def id2excepted_format(self, IDs, **kwargs):
        msg = ''
        for ID in IDs:
            information = {**self.id2information[ID]}
            if 'description' not in information:
                information['description'] = information['name']
            description = information['description']
            if self.max_description_len_on_prompt > 0:
                description = description.split(' ')

                if self.max_description_len_on_prompt < len(description):
                    description = description[:self.max_description_len_on_prompt]
                    description.append('...')
                else:
                    description = description[:self.max_description_len_on_prompt]
                description = ' '.join(description)

            information['description'] = description
            msg += f'<Object>{json.dumps(information)}</Object>\n'
        return msg

    def get_labels_of_indexes(self, indexes, **kwargs):
        return [self.informations_embeddings['idx2id'][i] for i in indexes]

    def __init__(self, name, data_path, data_file_name,
                 cache_folder=None,
                 PLM_name='all-MiniLM-L6-v2',
                 informations_embeddings_name='informations_embeddings-MiniLM.pt',
                 max_description_len_on_prompt=-1,
                 id2information_name='id2information.json',
                 **kwargs):
        super().__init__(name, data_path)
        self.data_file_name = data_file_name
        self.cache_folder = cache_folder
        self.embedding_model = None
        self.device = None
        self.rel2examples = None
        self.dataset = None
        self.id2information = None
        self.id2idxes = None
        self.informations_embeddings_name = informations_embeddings_name
        self.informations_embeddings = None
        self.PLM_name = PLM_name
        self.id2information_name = id2information_name
        self.max_description_len_on_prompt = max_description_len_on_prompt
        self.load_dataset()

    def load_dataset(self):
        self.dataset = json.load(open(os.path.join(self.data_path, self.data_file_name), 'r', encoding='utf-8'))
        self.rel2examples = json.load(open(os.path.join(self.data_path, 'rel2examples.json'), 'r', encoding='utf-8'))
        self.dataset = list(self.dataset.items())
        self.id2information = json.load(
            open(os.path.join(self.data_path, self.id2information_name), 'r', encoding='utf-8'))
        self.id2idxes = {k: i for i, (k, v) in enumerate(list(self.id2information.items()))}
        self.informations_embeddings = torch.load(os.path.join(self.data_path, self.informations_embeddings_name))

        self.attr2id_dict = {}
        for k, v in self.id2information.items():
            for vk, vv in v.items():
                if vk not in self.attr2id_dict:
                    self.attr2id_dict[vk] = {}
                self.attr2id_dict[vk][str(vv)] = k  # to id

    def get_label_information(self, label):
        return len(self.id2information)

    def __getitem__(self, item):
        if type(item) is int:
            return self.handle_data2dict(self.dataset[item])
        else:
            return [self[i] for i in range(len(self))[item]]

    def __len__(self):
        return len(self.dataset)

    def save_one_prediction(self, IO, current_data, prediction, LLM_response):
        saved_data = {
            'ID': current_data['ID'],
            'LLM_response': LLM_response,
            'prediction': prediction,
            'labels': current_data['labels']
        }
        IO.write(json.dumps(saved_data) + '\n')
        IO.flush()
        return saved_data

    def handle_result4LLM(self, LLM_response):
        objs = re.findall(r'<Object>(.*?)</Object>', LLM_response)
        objs = list(set(objs))
        objs.append(LLM_response)
        return objs

        # def get_indexes_of_labels(self, labels, pre_topk=1, **kwargs):
        #     if self.embedding_model is None:
        #         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #         self.embedding_model = SentenceTransformer(self.PLM_name, cache_folder=self.cache_folder).to(
        #             self.device)
        #         self.informations_embeddings['informations_embeddings'] = self.informations_embeddings[
        #             'informations_embeddings'].to(self.device)
        #
        #     idxs = []
        #     if not labels:
        #         # null
        #         return idxs
        #
        #     if labels[0] in self.id2idxes:
        #         for l in labels:
        #             idxs.append(self.id2idxes[l])
        #     else:
        #         labels_embedding = torch.stack(
        #             [self.embedding_model.encode(i, convert_to_tensor=True).to(self.device) for i in labels])
        #         s = labels_embedding @ self.informations_embeddings['informations_embeddings'].T
        #         eidxs = s.topk(pre_topk).indices.flatten().tolist()
        #         for eidx in eidxs:
        #             eid = self.informations_embeddings['idx2id'][eidx]
        #             if eid not in idxs:
        #                 idxs.append(self.id2idxes[eid])

        return idxs

    def get_label_number(self):
        return len(self.id2information)

    def handle_data2dict(self, data):
        hr, ts = data
        head, relation = eval(hr)

        head_information = self.id2information[head]

        ky = {
            'relation': relation,
            **head_information,
            'labels': ts,
            'examples': self.build_example_text(relation),
            'ID': hr,
        }

        return ky

    def build_example_text(self, rel):
        examples = self.rel2examples[rel]
        example_text = ''
        for h, ts in examples:
            example_text += 'Subject:\n'
            try:
                h_information = self.id2information[h]
            except:
                return ''
            example_text += f'\tname: {h_information["name"]}\n'
            example_text += f'\tdescription: {h_information["description"]}\n\n'

            example_text += f'Predicate: {rel}\n\n'

            example_text += 'Objects:\n'
            for t in ts:
                try:
                    t_information = self.id2information[t]
                except:
                    return ''
                t_kv = {**t_information}
                example_text += f"<Object>{json.dumps(t_kv)}</Object>\n"
            example_text += '\n'

        return example_text


class Reuters21578Dataset(MultiLabelDataset):
    def get_indexes_of_labels(self, labels, pre_topk=1, isLabels=False, **kwargs):
        self._load_embedding_model()

        idxs = []
        if not labels:
            # null
            return idxs

        if isLabels:
            for l in labels:
                if l not in idxs and l in self.attr2id_dict['name']:
                    idxs.append(self.id2idxes[self.attr2id_dict['name'][l]])
        else:
            labels_embedding = torch.stack(
                [self.embedding_model.encode(i, convert_to_tensor=True).to(self.device) for i in labels])
            s = labels_embedding @ self.informations_embeddings['informations_embeddings'].T
            eidxs = s.topk(pre_topk).indices.flatten().tolist()
            for eidx in eidxs:
                eid = self.informations_embeddings['idx2id'][eidx]
                if eid not in idxs:
                    idxs.append(self.id2idxes[eid])

        return idxs

    def id2excepted_format(self, IDs, **kwargs):
        msg = ''
        for ID in IDs:
            information = self.id2information[ID]
            msg += f'<Label>{information}</Label>\n'
        msg += '<Summary>' + '; '.join([self.id2information[ID] for ID in IDs]) + '</Summary>'
        return msg

    def get_labels_of_indexes(self, indexes, **kwargs):
        return [self.informations_embeddings['idx2id'][i] for i in indexes]

    def __init__(self, name, data_path, data_file_name,
                 cache_folder=None,
                 informations_embeddings_name='informations_embeddings-MiniLM.pt',
                 PLM_name='all-MiniLM-L6-v2',
                 examples_path='examples.json',
                 id2information_name='id2information.json',
                 **kwargs):
        super().__init__(name, data_path)
        self.examples = None
        self.data_file_name = data_file_name
        self.cache_folder = cache_folder
        self.embedding_model = None
        self.device = None
        self.dataset = None
        self.id2information = None
        self.id2idxes = None
        self.informations_embeddings = None
        self.informations_embeddings_name = informations_embeddings_name
        self.id2information_name= id2information_name
        self.PLM_name = PLM_name
        self.examples_path = examples_path
        self.load_dataset()

    def load_dataset(self):
        self.dataset = json.load(open(os.path.join(self.data_path, self.data_file_name), 'r', encoding='utf-8'))
        self.examples = json.load(open(os.path.join(self.data_path, self.examples_path), 'r', encoding='utf-8'))
        self.dataset = list(self.dataset.items())
        self.id2information = json.load(
            open(os.path.join(self.data_path, self.id2information_name), 'r', encoding='utf-8'))
        self.id2idxes = {k: i for i, (k, v) in enumerate(list(self.id2information.items()))}
        self.informations_embeddings = torch.load(os.path.join(self.data_path, self.informations_embeddings_name))

        # 对齐
        self.attr2id_dict = {}
        self.attr2id_dict['name'] = {}
        for k, v in self.id2information.items():
                self.attr2id_dict['name'][str(v)] = k  # to id

    def get_label_information(self, label):
        return len(self.id2information)

    def __getitem__(self, item):
        if type(item) is int:
            return self.handle_data2dict(self.dataset[item])
        else:
            return [self[i] for i in range(len(self))[item]]

    def __len__(self):
        return len(self.dataset)

    def save_one_prediction(self, IO, current_data, prediction, LLM_response):
        saved_data = {
            'ID': current_data['ID'],
            'LLM_response': LLM_response,
            'prediction': prediction,
            'labels': current_data['labels']
        }
        IO.write(json.dumps(saved_data) + '\n')
        IO.flush()
        return saved_data

    def handle_result4LLM(self, LLM_response):
        labels = set()

        summary_result = re.findall(r'<Summary>(.*)</Summary>', LLM_response)
        for r in summary_result:
            for c in r.split(';'):
                c = c.strip().replace('.', '')
                labels.add(c)

        code_result = re.findall(r'<Label>(.*?)</Label>', LLM_response)
        for r in code_result:
            for c in r.split(';'):
                c = c.strip().replace('.', '')
                labels.add(c)

        if len(labels) == 0:
            return None
        else:
            return list(labels)

    def get_label_number(self):
        return len(self.id2information)

    def handle_data2dict(self, data):
        ID, msg = data
        ky = {
            **msg,
            'examples': self.build_example_text(),
            'ID': ID,
        }

        return ky

    def build_example_text(self):
        examples = self.examples
        example_text = ''
        for idx, (ID, msg) in enumerate(examples.items()):
            example_text += f'Example {idx + 1}:\n'
            example_text += 'Text:\n'
            example_text += msg['text'] + '\n'

            example_text += 'Prediction:\n'
            for l in msg['labels']:
                example_text += f"<Label>{l}</Label>\n"
            example_text += f'<Summary>{"; ".join(msg["labels"])}</Summary>\n'
            example_text += '\n'

        return example_text

    def _str_align(self, labels, pre_topk=1, isLabels=False, mode='vec_sim', **kwargs):
        indexes = []

        for label in labels:
            value = label
            if value in self.attr2id_dict['name']:
                vid = self.attr2id_dict['name'][value]
                idx = self.id2idxes[vid]
                if idx not in indexes:
                    indexes.append(idx)
        return indexes

def load_dataset(dataset, **kwargs) -> MultiLabelDataset:
    if dataset == 'codiesp':
        return CodiespDataset(**{'name': dataset, **kwargs})
    elif dataset == 'WN18RR':
        return WN18RRDataset(**{'name': dataset, **kwargs})
    elif dataset == 'FB15k237':
        return FB15k237Dataset(**{'name': dataset, **kwargs})
    elif dataset == 'Reuters21578':
        return Reuters21578Dataset(**{'name': dataset, **kwargs})
