#!/bin/sh

for subtask in 'dp' 'bd' 'mfw'
do
    gft_summary --data P:cote,$subtask --splits train,test,test
done

