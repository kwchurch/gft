#!/bin/sh

echo hostname = `hostname`

for model in textattack/bert-base-uncased-imdb lvwerra/bert-imdb textattack/distilbert-base-uncased-imdb textattack/roberta-base-imdb aychang/roberta-base-imdb lvwerra/distilbert-imdb textattack/albert-base-v2-imdb ykacer/bert-base-cased-imdb-sequence-classification nateraw/bert-base-uncased-imdb aypan17/roberta-base-imdb allenai/dsp_roberta_base_tapt_imdb_70000 textattack/xlnet-base-cased-imdb Lumos/imdb4 abhishek/autonlp-imdb_sentiment_classification-31154 w11wo/javanese-bert-small-imdb-classifier yosemite/autonlp-imdb-sentiment-analysis-english-470512388 abhishek/autonlp-imdb_eval-71421 lannelin/bert-imdb-1hidden juliensimon/autonlp-imdb-demo-hf-16622775 Sakil/imdbsentdistilbertmodel ZiweiG/ziwei-bert-imdb ZiweiG/ziwei-bertimdb-prob lschneidpro/distilbert_uncased_imdb 

	     do
    echo MODEL = $model
gft_predict --model H:$model \
    --data H:imdb \
    --split test \
    --eqn 'classify: label ~ text'
done



# w11wo/javanese-gpt2-small-imdb textattack/bert-base-uncased-imdb lvwerra/bert-imdb textattack/distilbert-base-uncased-imdb mrm8488/t5-base-finetuned-imdb-sentiment textattack/roberta-base-imdb lvwerra/gpt2-imdb aychang/roberta-base-imdb lvwerra/distilbert-imdb textattack/albert-base-v2-imdb huggingface-course/distilbert-base-uncased-finetuned-imdb yxchar/tlm-imdb-small-scale ykacer/bert-base-cased-imdb-sequence-classification nateraw/bert-base-uncased-imdb mrm8488/t5-small-finetuned-imdb-sentiment aypan17/roberta-base-imdb allenai/dsp_roberta_base_tapt_imdb_70000 textattack/xlnet-base-cased-imdb Lumos/imdb4 abhishek/autonlp-imdb_sentiment_classification-31154 w11wo/javanese-bert-small-imdb-classifier yosemite/autonlp-imdb-sentiment-analysis-english-470512388 abhishek/autonlp-imdb_eval-71421 lannelin/bert-imdb-1hidden juliensimon/autonlp-imdb-demo-hf-16622775 keras-io/bidirectional-lstm-imdb Sakil/imdbsentdistilbertmodel ZiweiG/ziwei-bert-imdb ZiweiG/ziwei-bertimdb-prob lschneidpro/distilbert_uncased_imdb 
