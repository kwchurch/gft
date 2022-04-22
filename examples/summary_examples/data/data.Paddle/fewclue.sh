#!/bin/sh

# for split in 'train_0', 'train_1', 'train_2', 'train_3', 'train_4', 'train_few_all', 'dev_0', 'dev_1', 'dev_2', 'dev_3', 'dev_4', 'dev_few_all', 'unlabeled', 'test', 'test_public'

for subtask in 'bustm' 'chid' 'iflytek' 'tnews' 'eprstmt' 'ocnli' 'csldcp' 'cluewsc' 'csl'
	       do
gft_summary --data P:fewclue,$subtask --split train_0,train_1,train_2,train_3,train_4,train_few_all,dev_0,dev_1,dev_2,dev_3,dev_4,dev_few_all,unlabeled,test,test_public
done

