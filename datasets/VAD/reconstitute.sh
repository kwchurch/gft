#!/bin/sh

# set VAD to your copy of the NRC-VAD Lexicon (from https://saifmohammad.com/WebPages/nrc-vad.html)

# Run this to create three files VAD.train, VAD.val and VAD.test

# Replace VAD_splits.txt with VAD.small_splits.txt and VAD.tiny_splits.txt to create
#    VAD.small.train, VAD.small.val and VAD.small.test
#    VAD.tiny.train, VAD.tiny.val and VAD.tiny.test

tr '\t' ',' < $VAD |
awk -F, 'NR == 1 {header=$0; next}; 
          {VAD[$1]=$0}; 
        END {FS="\t"; while(getline < splits > 0) {
	    	      		    if(killroy[$2]++==0) print header > $2; 
				    print VAD[$1] >> $2}}'  splits=VAD_splits.txt
