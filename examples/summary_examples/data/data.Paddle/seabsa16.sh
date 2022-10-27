#!/bin/sh

gft_summary --data P:seabsa16,phns --splits train,test,test --split test
gft_summary --data P:seabsa16,came --splits train,test,test --split test

