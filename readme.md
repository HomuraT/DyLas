

[TOC]

# Supplemental Material for DyLas: A Dynamic Label Alignment Strategy for Large-Scale Multi-Label Text Classification

This is the supplemental material of our paper for reproducibility, including:

# Requirements

```
openai=1.9.0
scikit-learn=1.4.0
scipy=1.12.0
sentence-transformers=2.3.1
torch=1.13.0+cu116
chromadb=0.4.24
wandb=0.16.4
```

# Datasets

For datasets WN18RR and FB25k-127, [SimKGC](https://github.com/intfloat/simkgc) provides their resources for downloading.



Codiesp is available in this url: https://zenodo.org/records/3837305#.XsZFoXUzZpg

​	Label description url: https://github.com/StefanoTrv/simple_icd_10_CM/blob/master/data/icd10cm-order-Jan-2021.txt



Reuters21578 is available in this url: https://www.kaggle.com/datasets/nltkdata/reuters/data

## Directory structure

```ymal
dataset
|-codiesp
  |- final_dataset_v4_to_publish
    |- background
    |- dev
    |- test
    |- train
    |- README.txt
  |- icd10cm-order-Jan-2021.txt
|- WN18RR
  |- wordnet-mlj12-definitions.txt
  |- test.txt
  |- valid.txt
  |- train.txt
  |- entities.dict
  |- relations.dict
|- FB15k237
  |- valid.txt
  |- train.txt
  |- test.txt
  |- FB15k_mid2description.txt
  |- FB15k_mid2name.txt
|- Reuters21578
  |- reuters
    |- ...
  |- test
    |- ...
  |- training
    |- ...
  |- cats.txt
  |- README
  |- stopwords
```

## Data Preprocessing

```shell
python dataset_preprocessing.py --dataset codiesp --data_path ./dataset/codiesp

python dataset_preprocessing.py --dataset WN18RR --data_path dataset/WN18RR

python dataset_preprocessing.py --dataset FB15k237 --data_path dataset/FB15k237

python dataset_preprocessing.py --dataset Reuters21578 --data_path dataset/Reuters21578
```

## Create Label Embeddings

Execute the jupyter file in dataset path to create the label embeddings, such as `dataset/codiesp/01_codiesp_information_embedding.ipynb` .

In the Supplemental Material, we provide the label embeddings on the Reuters21578 dataset.

# Predict

## Set api key

To call the API for LLMs, you need to set api-key in the file `./llm/resources/ampi.json`

```json
{
     "gpt35": {
          "url": "https://api.openai.com/v1",
          "api_key": "your-key",
          "model": "gpt-3.5-turbo",
          "history": []
     },"gpt4": {
          "url": "https://api.openai.com/v1",
          "api_key": "your-key",
          "model": "gpt-4",
          "history": []
     },"baseline_ollama_llama3_1_70b": {
          "url": "http://localhost:11434/api/chat",
          "api_key": "ollama",
          "model": "llama3.1:70b",
          "history": []
     }
}
```

## Run (llama3.1-70b)

```shell
# FB15k237
python llm_predict.py --dataset FB15k237 --data_path dataset/FB15k237 --prediction_path ./predictions/FB15k237/baseline.txt --prompt_path prompts/FB15k237.json --prompt_id directly_predict --api_name baseline_ollama_llama3_1_70b --label_description_path dataset/FB15k237/relations.dict --data_file_name subset_500.json --id2information_name id2information.json

# Reuters21578
python llm_predict.py --dataset Reuters21578 --data_path dataset/Reuters21578 --prompt_path prompts/Reuters21578.json --prompt_id directly_predict --label_description_path dataset/Reuters21578/relations.dict --data_file_name subset_500.json --api_name baseline_ollama_llama3_1_70b --examples_path examples.json --id2information_name id2information.json --prediction_path ./predictions/Reuters21578/baseline.txt

# WN18RR
python llm_predict.py --dataset WN18RR --data_path dataset/WN18RR --prediction_path ./predictions/WN18RR/baseline.txt --prompt_path prompts/WN18RR.json --prompt_id directly_predict --label_description_path filterd_labels_triple_description_question_50.json --data_file_name subset_500.json --api_name baseline_ollama_llama3_1_70b

# codiesp
python llm_predict.py --dataset codiesp --data_path ./dataset/codiesp/ --data_file_name test.json --label_description_path label_description.json --handle_mode desc_dict --prediction_path ./predictions/codiesp/baseline.txt
```

## DyLas

```shell
# FB15k237
python llm_predict_by_old_prediction.py --dataset FB15k237 --data_path dataset/FB15k237 --api_name baseline_ollama_llama3_1_70b --label_description_path dataset/FB15k237/relations.dict --data_file_name subset_500.json --id2information_name id2information.json --old_prediction_path ./predictions/FB15k237/baseline.txt --prompt_path prompts/FB15k237_by_op.json --prompt_id directly_predict_wo_example --api_name baseline_ollama_llama3_1_70b --label_description_path dataset/FB15k237/relations.dict --data_file_name subset_500.json --pre_topk 1 --prediction_path ./predictions/FB15k237/baseline_DyLas.txt --id2information_name id2information.json --PLM_name BAAI/bge-m3 --informations_embeddings_name informations_embeddings_bge.pt

# Reuters21578
python llm_predict_by_old_prediction.py --dataset Reuters21578 --data_path dataset/Reuters21578 --prompt_path prompts/Reuters21578_by_op.json --prompt_id directly_predict --label_description_path dataset/Reuters21578/relations.dict --data_file_name top10_subset_500.json --api_name baseline_ollama_llama3_1_70b --examples_path examples.json --id2information_name id2information.json  --old_prediction_path ./predictions/Reuters21578/baseline.txt --informations_embeddings_name informations_embeddings_MiniLM.pt --PLM_name all_MiniLM-L6-v2 --prediction_path ./predictions/Reuters21578/baseline_DyLas.txt

# WN18RR
python llm_predict_by_old_prediction.py --dataset WN18RR --data_path dataset/WN18RR --api_name baseline_ollama_llama3_1_70b --label_description_path dataset/WN18RR/relations.dict --data_file_name subset_500.json --id2information_name id2information.json --old_prediction_path ./predictions/WN18RR/baseline.txt --prompt_path prompts/WN18RR_by_op.json --prompt_id directly_predict --api_name baseline_ollama_llama3_1_70b --label_description_path filterd_labels_triple_description_question_50.json --data_file_name subset_500.json --pre_topk 1 --id2information_name id2information.json --PLM_name BAAI/bge-m3 --informations_embeddings_name sub_informations_embeddings_bge.pt --prediction_path ./predictions/WN18RR/baseline_DyLas.txt

# codiesp
python llm_predict_by_old_prediction.py --dataset codiesp --data_path ./dataset/codiesp/ --data_file_name test.json --label_description_path label_description.json --handle_mode desc_dict --old_prediction_path ./predictions/codiesp/baseline.txt --prompt_path prompts/codiesp_by_op.json --prompt_id directly_predict --api_name baseline_ollama_llama3_1_70b --PLM_name all_MiniLM-L6-v2 --informations_embeddings_name informations_embeddings_MiniLM.pt --prediction_path ./predictions/codiesp/baseline_DyLas.txt
```

# Evaluate (llama3.1-70b)

## Reuters21578

HA:

```shell
HF_ENDPOINT=https://hf-mirror.com python evaluate_llm_results_by_modes.py --dataset Reuters21578 --data_path dataset/Reuters21578 --label_description_path dataset/Reuters21578/relations.dict --data_file_name subset_500.json --examples_path examples.json --id2information_name id2information.json --prediction_path ./predictions/Reuters21578/baseline.txt --mode str_align
```

HA+DyLas

```shell
HF_ENDPOINT=https://hf-mirror.com python evaluate_llm_results_by_modes.py --dataset Reuters21578 --data_path dataset/Reuters21578 --label_description_path dataset/Reuters21578/relations.dict --data_file_name subset_500.json --examples_path examples.json --id2information_name id2information.json --prediction_path ./predictions/Reuters21578/baseline_DyLas_HA.txt --mode str_align 
```

SA:

```shell
HF_ENDPOINT=https://hf-mirror.com python evaluate_llm_results_by_modes.py --dataset Reuters21578 --data_path dataset/Reuters21578 --label_description_path dataset/Reuters21578/relations.dict --data_file_name subset_500.json --examples_path examples.json --id2information_name id2information.json --prediction_path ./predictions/Reuters21578/baseline.txt  --mode vec_sim --mode vec_sim --PLM_name all_MiniLM-L6-v2  --informations_embeddings_name informations_embeddings_MiniLM.pt
```
SA+DyLas

```shell
HF_ENDPOINT=https://hf-mirror.com python evaluate_llm_results_by_modes.py --dataset Reuters21578 --data_path dataset/Reuters21578 --label_description_path dataset/Reuters21578/relations.dict --data_file_name subset_500.json --examples_path examples.json --id2information_name id2information.json --prediction_path ./predictions/Reuters21578/baseline_DyLas_HA.txt  --mode vec_sim 
```


## codiesp

HA:

```shell
HF_ENDPOINT=https://hf-mirror.com python evaluate_llm_results_by_modes.py --dataset codiesp --data_path ./dataset/codiesp/ --data_file_name test.json --label_description_path label_description.json --handle_mode desc_dict --prediction_path ./predictions/codiesp/baseline.txt --mode str_align
```
HA+DyLas:

```shell
HF_ENDPOINT=https://hf-mirror.com python evaluate_llm_results_by_modes.py --dataset codiesp --data_path ./dataset/codiesp/ --data_file_name test.json --label_description_path label_description.json --handle_mode desc_dict --prediction_path ./predictions/codiesp/baseline_DyLas_HA.txt --mode str_align
```

SA:

```shell
HF_ENDPOINT=https://hf-mirror.com python evaluate_llm_results_by_modes.py --dataset codiesp --data_path ./dataset/codiesp/ --data_file_name test.json --label_description_path label_description.json --handle_mode desc_dict --prediction_path ./predictions/codiesp/baseline.txt --mode vec_sim --PLM_name BAAI/bge-m3   --informations_embeddings_name informations_embeddings_bge.pt
```

SA+DyLas:

```shell
HF_ENDPOINT=https://hf-mirror.com python evaluate_llm_results_by_modes.py --dataset codiesp --data_path ./dataset/codiesp/ --data_file_name test.json --label_description_path label_description.json --handle_mode desc_dict --prediction_path ./predictions/codiesp/baseline_DyLas.txt --mode vec_sim --PLM_name all_MiniLM-L6-v2   --informations_embeddings_name informations_embeddings_MiniLM.pt
```

## FB15k237

HA:

```shell
HF_ENDPOINT=https://hf-mirror.com python evaluate_llm_results_by_modes.py --dataset FB15k237 --data_path dataset/FB15k237 --data_file_name subset_500.json --id2information_name id2information.json --mode str_align --prediction_path ./predictions/FB15k237/baseline.txt
```

HA+DyLas

```shell
HF_ENDPOINT=https://hf-mirror.com python evaluate_llm_results_by_modes.py --dataset FB15k237 --data_path dataset/FB15k237 --data_file_name subset_500.json --id2information_name id2information.json --prediction_path ./predictions/FB15k237/baseline_DyLas_HA.txt --mode str_align
```

SA:

```shell
HF_ENDPOINT=https://hf-mirror.com python evaluate_llm_results_by_modes.py --dataset FB15k237 --data_path dataset/FB15k237 --data_file_name subset_500.json --id2information_name id2information.json --mode str_align --prediction_path ./predictions/FB15k237/baseline.txt --mode vec_sim --PLM_name all_MiniLM-L6-v2  --informations_embeddings_name informations_embeddings_MiniLM.pt
```

SA+DyLas

```shell
HF_ENDPOINT=https://hf-mirror.com python evaluate_llm_results_by_modes.py --dataset FB15k237 --data_path dataset/FB15k237 --data_file_name subset_500.json --id2information_name id2information.json --prediction_path ./predictions/FB15k237/baseline_DyLas_SA.txt --mode vec_sim --PLM_name BAAI/bge-m3   --informations_embeddings_name informations_embeddings_bge.pt
```



## WN18RR

HA:

```shell
HF_ENDPOINT=https://hf-mirror.com python evaluate_llm_results_by_modes.py --dataset WN18RR --data_path dataset/WN18RR --data_file_name subset_500.json --label_description_path filterd_labels_triple_description_question_50.json --prediction_path ./predictions/WN18RR/baseline.txt --mode str_align
```

HA+DyLas:

```shell
HF_ENDPOINT=https://hf-mirror.com python evaluate_llm_results_by_modes.py --dataset WN18RR --data_path dataset/WN18RR --data_file_name subset_500.json --label_description_path filterd_labels_triple_description_question_50.json --prediction_path ./predictions/WN18RR/baseline_DyLas_HA.txt --mode str_align
```

SA:

```shell
HF_ENDPOINT=https://hf-mirror.com python evaluate_llm_results_by_modes.py --dataset WN18RR --data_path dataset/WN18RR --data_file_name subset_500.json --label_description_path filterd_labels_triple_description_question_50.json --prediction_path ./predictions/WN18RR/baseline.txt --mode vec_sim --PLM_name all_MiniLM-L6-v2  --informations_embeddings_name informations_embeddings_MiniLM.pt
```

SA+DyLas:

```shell
HF_ENDPOINT=https://hf-mirror.com python evaluate_llm_results_by_modes.py --dataset WN18RR --data_path dataset/WN18RR --data_file_name subset_500.json --label_description_path filterd_labels_triple_description_question_50.json --prediction_path ./predictions/WN18RR/baseline_DyLas_SA.txt --mode vec_sim --PLM_name all_MiniLM-L6-v2  --informations_embeddings_name informations_embeddings_MiniLM.pt
```

# Results(llama3.1-70b)

**Note that, for easily reproduce the results, we provide the final results in the directory `./predictions` and the final results in the directory `./predictions`.**