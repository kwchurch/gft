---
license: apache-2.0
tags:
- automatic-speech-recognition
- timit_asr
- generated_from_trainer
model-index:
- name: wav2vec2-base.timit_asr.32
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# wav2vec2-base.timit_asr.32

This model is a fine-tuned version of [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) on the TIMIT_ASR - NA dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3392
- Wer: 0.2167

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 32
- eval_batch_size: 1
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 1000
- num_epochs: 20.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer    |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| 3.2001        | 0.69  | 100  | 3.1346          | 1.0    |
| 2.9621        | 1.38  | 200  | 2.9686          | 1.0    |
| 2.9307        | 2.07  | 300  | 2.9096          | 1.0    |
| 2.6335        | 2.76  | 400  | 2.5031          | 1.0    |
| 1.2161        | 3.45  | 500  | 1.0692          | 0.6701 |
| 0.5591        | 4.14  | 600  | 0.6256          | 0.4506 |
| 0.4672        | 4.83  | 700  | 0.4926          | 0.3807 |
| 0.5269        | 5.52  | 800  | 0.4421          | 0.3547 |
| 0.2858        | 6.21  | 900  | 0.4367          | 0.3306 |
| 0.2026        | 6.9   | 1000 | 0.3824          | 0.3052 |
| 0.2029        | 7.59  | 1100 | 0.3823          | 0.3028 |
| 0.2649        | 8.28  | 1200 | 0.3524          | 0.2796 |
| 0.2036        | 8.97  | 1300 | 0.3817          | 0.2777 |
| 0.1246        | 9.66  | 1400 | 0.3503          | 0.2690 |
| 0.101         | 10.34 | 1500 | 0.3596          | 0.2608 |
| 0.1595        | 11.03 | 1600 | 0.3444          | 0.2499 |
| 0.1218        | 11.72 | 1700 | 0.3571          | 0.2468 |
| 0.0757        | 12.41 | 1800 | 0.3430          | 0.2413 |
| 0.0607        | 13.1  | 1900 | 0.3533          | 0.2411 |
| 0.1129        | 13.79 | 2000 | 0.3419          | 0.2284 |
| 0.0752        | 14.48 | 2100 | 0.3512          | 0.2319 |
| 0.0626        | 15.17 | 2200 | 0.3509          | 0.2300 |
| 0.0441        | 15.86 | 2300 | 0.3256          | 0.2268 |
| 0.0744        | 16.55 | 2400 | 0.3357          | 0.2211 |
| 0.0811        | 17.24 | 2500 | 0.3435          | 0.2241 |
| 0.0454        | 17.93 | 2600 | 0.3437          | 0.2222 |
| 0.0316        | 18.62 | 2700 | 0.3380          | 0.2190 |
| 0.0571        | 19.31 | 2800 | 0.3399          | 0.2182 |
| 0.0667        | 20.0  | 2900 | 0.3392          | 0.2167 |


### Framework versions

- Transformers 4.15.0
- Pytorch 1.9.1+cu102
- Datasets 1.18.0
- Tokenizers 0.10.1
