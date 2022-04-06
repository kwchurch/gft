#!/bin/sh

echo hostname = `hostname`

for model in  uer/roberta-base-finetuned-chinanews-chinese # uer/roberta-base-finetuned-dianping-chinese uer/roberta-base-finetuned-jd-full-chinese uer/roberta-base-finetuned-jd-binary-chinese saattrupdan/verdict-classifier fnlp/cpt-large symanto/xlm-roberta-base-snli-mnli-anli-xnli ethanyt/guwen-cls uer/roberta-base-finetuned-ifeng-chinese ethanyt/guwen-sent liam168/c2-roberta-base-finetuned-dianping-chinese liam168/c4-zh-distilbert-base-uncased tuhailong/cross-encoder-bert-base kyleinincubated/autonlp-cat33-624317932 Herais/pred_genre Herais/pred_timeperiod celtics1863/env-bert-topic kyleinincubated/autonlp-cat333-624217911 ysslang/autonlp-test-459011902 celtics1863/env-bert-cls-chinese

	     do
gft_eval --model H:$model \
    --data P:dureader_yesno \
    --split test \
    --eqn 'classify: labels ~ question + answer' \
    --do_not_catch_errors
done

# cannot find tokenizer for fnlp/cpt-base
