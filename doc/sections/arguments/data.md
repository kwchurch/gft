# Data

The following will produce a long list of datasets that are available on
(H) <a href="https://huggingface.co/datasets">HuggingFace</a> and (P) <a
href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/datasets">PaddleNLP</a>.

```sh
gft_summary --data H:__contains__ --topn 10000000 --fast_mode
gft_summary --data P:__contains__ --topn 10000000 --fast_mode
```

In addition to these datasets, one can also create models on the local disk with <a href="../functions/gft_fit.md">gft_fit</a> (or any other method).
<p>
Many of <i>gft</i> functions take a --data argument.  The value of this argument is typically: \<data_provider\>:\<data_key\>, where \<data_provider\> is a hub, H or P, and data_key is the name
of a dataset supported by the hub.




