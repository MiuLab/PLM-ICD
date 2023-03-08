# PLM-ICD: Automatic ICD Coding with Pretrained Language Models
[Paper](https://aclanthology.org/2022.clinicalnlp-1.2/)

Source code for our ClinicalNLP 2022 paper *PLM-ICD: Automatic ICD Coding with Pretrained Language Models*

    @inproceedings{huang-etal-2022-plm,
        title = "{PLM}-{ICD}: Automatic {ICD} Coding with Pretrained Language Models",
        author = "Huang, Chao-Wei and Tsai, Shang-Chi and Chen, Yun-Nung",
        booktitle = "Proceedings of the 4th Clinical Natural Language Processing Workshop",
        month = jul,
        year = "2022",
        address = "Seattle, WA",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2022.clinicalnlp-1.2",
        pages = "10--20",
    }


## Requirements
* Python >= 3.6
* Install the required Python packages with `pip3 install -r requirements.txt`

## Dataset
Unfortunately, we are not allowed to redistribute the MIMIC dataset.
Please follow the instructions from [caml-mimic](https://github.com/jamesmullenbach/caml-mimic) to preprocess the MIMIC-2 and MIMIC-3 dataset and place the files under `data/mimic2` and `data/mimic3` respectively.

## How to run
### Pretrained LMs
Please download the pretrained LMs you want to use from the following link:
- [BioLM](https://github.com/facebookresearch/bio-lm): RoBERTa-PM models
- [BioBERT](https://github.com/dmis-lab/biobert)
- [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract): you can also set `--model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` when training the model, the script will download the checkpoint automatically.

### Trained Models
You can also download our [trained models](https://drive.google.com/drive/folders/1oJLgLKu_NZxsSTXU9uFVehxXXJYzTalO?usp=sharing) to skip the training part. We provide 3 trained models:
- [Trained on MIMIC-3 full](https://drive.google.com/drive/folders/1SXlyh4ydRqlLwed_tiBA2mNCDjVll6gD?usp=sharing)
- [Trained on MIMIC-3 50](https://drive.google.com/drive/folders/12xRNiaXbwmrAcqzkUo96EpopBuICnWqR?usp=sharing)
- [Trained on MIMIC-2](https://drive.google.com/drive/folders/1tmopSwLccrBpHCoalAz-oRKAlxBvyF0H?usp=sharing)

### Training
1. `cd src`
2. Run the following command to train a model on MIMIC-3 full.
```
python3 run_icd.py \
    --train_file ../data/mimic3/train_full.csv \
    --validation_file ../data/mimic3/dev_full.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../models/RoBERTa-base-PM-M3-Voc-distill-align-hf \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 20 \
    --num_warmup_steps 2000 \
    --output_dir ../models/roberta-mimic3-full \
    --model_type roberta \
    --model_mode laat
```

### Notes
- If you would like to train BERT-based or Longformer-base models, please set `--model_type [bert|longformer]`.
- If you would like to train models on MIMIC-3 top-50, please set `--code_50 --code_file ../data/mimic3/ALL_CODES_50.txt`
- If you would like to train models on MIMIC-2, please set `--code_file ../data/mimic2/ALL_CODES.txt`

### Inference
1. `cd src`
2. Run the following command to evaluate a model on the test set of MIMIC-3 full.
```
python3 run_icd.py \
    --train_file ../data/mimic3/train_full.csv \
    --validation_file ../data/mimic3/test_full.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../models/roberta-mimic3-full \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 0 \
    --output_dir ../models/roberta-mimic3-full \
    --model_type roberta \
    --model_mode laat
```
