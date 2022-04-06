#!/bin/sh



for subtask in 'cola' 'sst-2' 'sts-b' 'qqp' 'mnli' 'qnli' 'rte' 'wnli' 'mrpc'
	       do
		   gft_summary --data P:glue,$subtask
done

gft_summary --data P:glue,mnli --splits train,dev_matched,test_matched
gft_summary --data P:glue,mnli --splits train,dev_mismatched,test_mismatched
