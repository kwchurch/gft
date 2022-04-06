# Model

The following will produce a long list of models that are available on
(H) <a href="https://huggingface.co/models">HuggingFace</a> and (P) <a
href="https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html">PaddleNLP</a>.

```sh
gft_summary --model H:__contains__ --topn 10000000 --fast_mode
gft_summary --model P:__contains__ --topn 10000000 --fast_mode
```

In addition to these models, one can also create models on the local disk with <a href="../functions/gft_fit.md">gft_fit</a> (or any other method).
<p>
Many of <i>gft</i> functions take a --model argument.  The value of this argument is typically: <model_provider>:<model_key>, where <model_provider> is a hub, H or P, and model_key is the name
of a model supported by the hub.




