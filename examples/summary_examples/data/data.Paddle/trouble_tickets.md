# issues discovered with regression testing (gft_summary on Paddle datasets)

<h1>Ticket 1: 3/25/2022 Unresolved</h2>

```sh
gft_summary --data P:fewclue,cluewsc --do_not_catch_errors
```

The issue is in fewclue_cluewsc.tar.gz.  The corresponding json file is empty.
That signals an error.

```sh
# ValueError: No instances were read from the given filepath /mnt/home/kwc/.paddlenlp/datasets/FewCLUE/fewclue_cluewsc/unlabeled.json. Is the path correct?
```

```sh
ls -l /mnt/home/kwc/.paddlenlp/datasets/FewCLUE/*/unlabeled.json
-rw-r--r-- 1 kwc svail   381544 May 20  2021 /mnt/home/kwc/.paddlenlp/datasets/FewCLUE/fewclue_bustm/unlabeled.json
-rw-r--r-- 1 kwc svail  3467354 May 20  2021 /mnt/home/kwc/.paddlenlp/datasets/FewCLUE/fewclue_chid/unlabeled.json
-rw-r--r-- 1 kwc svail        0 May 20  2021 /mnt/home/kwc/.paddlenlp/datasets/FewCLUE/fewclue_cluewsc/unlabeled.json
-rw-r--r-- 1 kwc svail 10542231 May 20  2021 /mnt/home/kwc/.paddlenlp/datasets/FewCLUE/fewclue_csldcp/unlabeled.json
-rw-r--r-- 1 kwc svail 16645064 May 20  2021 /mnt/home/kwc/.paddlenlp/datasets/FewCLUE/fewclue_csl/unlabeled.json
-rw-r--r-- 1 kwc svail  3295268 May 20  2021 /mnt/home/kwc/.paddlenlp/datasets/FewCLUE/fewclue_eprstmt/unlabeled.json
-rw-r--r-- 1 kwc svail  6113303 May 20  2021 /mnt/home/kwc/.paddlenlp/datasets/FewCLUE/fewclue_iflytek/unlabeled.json
-rw-r--r-- 1 kwc svail  5757559 May 20  2021 /mnt/home/kwc/.paddlenlp/datasets/FewCLUE/fewclue_ocnli/unlabeled.json
-rw-r--r-- 1 kwc svail  3133192 May 20  2021 /mnt/home/kwc/.paddlenlp/datasets/FewCLUE/fewclue_tnews/unlabeled.json
```

