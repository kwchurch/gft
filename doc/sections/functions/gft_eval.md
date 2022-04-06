# gft_eval

In <a href="../../../examples/eval_examples/model.HuggingFace/language/data.HuggingFace/their_models/glue/mrpc.sh">$gft/examples/eval_examples/model.HuggingFace/language/data.HuggingFace/their_models/glue/mrpc.sh</a>
there is a shell script that looks like this:

```sh
for model in textattack/bert-base-uncased-MRPC \
    textattack/roberta-base-MRPC M-FAC/bert-tiny-finetuned-mrpc \
    textattack/albert-base-v2-MRPC ajrae/bert-base-uncased-finetuned-mrpc \
    textattack/distilbert-base-uncased-MRPC howey/roberta-large-mrpc \
    textattack/distilbert-base-cased-MRPC \
    textattack/xlnet-large-cased-MRPC \
    gchhablani/bert-base-cased-finetuned-mrpc  \
    ji-xin/roberta_base-MRPC-two_stage shivangi/MRPC_64_128_output  \
    ntrnghia/mrpc_vn shivangi/MRPC_output  \
    yoshitomo-matsubara/bert-base-uncased-mrpc  \
    yoshitomo-matsubara/bert-base-uncased-mrpc_from_bert-large-uncased-mrpc  \
    yoshitomo-matsubara/bert-large-uncased-mrpc  \
    howey/bert-base-uncased-mrpc Alireza1044/albert-base-v2-mrpc  \
    AdapterHub/bert-base-uncased-pf-mrpc  \
    mrm8488/deberta-v3-small-finetuned-mrpc  \
    textattack/xlnet-base-cased-MRPC Maelstrom77/bert-base-uncased-MRPC  \
    Maelstrom77/roberta-large-mrpc Riad/finetuned-bert-mrpc  \
    M-FAC/bert-mini-finetuned-mrpc NDugar/finetuned-bert-mrpc  \
do
    gft_eval --data H:glue,mrpc --eqn 'classify: label ~ sentence1 + sentence2' --split val  --model H:$model --metric H:glue,mrpc
done
```

The results are in <a href="../../../examples/eval_examples/model.HuggingFace/language/data.HuggingFace/their_models/glue/mrpc.out">$gft/examples/eval_examples/model.HuggingFace/language/data.HuggingFace/their_models/glue/mrpc.out</a>.
Those results can be summarized into a table like this:

<table>
<tr><th>Model</th><th>Accuracy</th><th>f1</th><th>Seconds</th><th>Downloads</th></tr>
<tr><td>textattack/roberta-base-MRPC</td><td>0.912</td><td>0.936</td><td>3.18</td><td>1623</td></tr>
<tr><td>textattack/albert-base-v2-MRPC</td><td>0.897</td><td>0.926</td><td>3.39</td><td>175</td></tr>
<tr><td>mrm8488/deberta-v3-small-finetuned-mrpc</td><td>0.892</td><td>0.923</td><td>2.39</td><td>30</td></tr>
<tr><td>textattack/xlnet-base-cased-MRPC</td><td>0.890</td><td>0.920</td><td>3.52</td><td>96</td></tr>
<tr><td>yoshitomo-matsubara/bert-large-uncased-mrpc</td><td>0.880</td><td>0.916</td><td>6.26</td><td>59</td></tr>
<tr><td>Maelstrom77/roberta-large-mrpc</td><td>0.880</td><td>0.913</td><td>6.31</td><td>38</td></tr>
<tr><td>howey/roberta-large-mrpc</td><td>0.880</td><td>0.913</td><td>5.33</td><td>62</td></tr>
<tr><td>textattack/bert-base-uncased-MRPC</td><td>0.877</td><td>0.913</td><td>4.34</td><td>10133</td></tr>
<tr><td>Maelstrom77/bert-base-uncased-MRPC</td><td>0.877</td><td>0.913</td><td>2.83</td><td>38</td></tr>
<tr><td>yoshitomo-matsubara/bert-base-uncased-mrpc_from_bert-large-uncased-mrpc</td><td>0.875</td><td>0.909</td><td>2.84</td><td>54</td></tr>
<tr><td>textattack/xlnet-large-cased-MRPC</td><td>0.875</td><td>0.913</td><td>8.17</td><td>97</td></tr>
<tr><td>yoshitomo-matsubara/bert-base-uncased-mrpc</td><td>0.873</td><td>0.907</td><td>3.03</td><td>52</td></tr>
<tr><td>Riad/finetuned-bert-mrpc</td><td>0.868</td><td>0.908</td><td>2.87</td><td>26</td></tr>
<tr><td>Alireza1044/albert-base-v2-mrpc</td><td>0.863</td><td>0.901</td><td>3.35</td><td>62</td></tr>
<tr><td>gchhablani/bert-base-cased-finetuned-mrpc</td><td>0.860</td><td>0.903</td><td>3.04</td><td>52</td></tr>
<tr><td>textattack/distilbert-base-uncased-MRPC</td><td>0.858</td><td>0.903</td><td>1.99</td><td>108</td></tr>
<tr><td>ajrae/bert-base-uncased-finetuned-mrpc</td><td>0.858</td><td>0.900</td><td>3.11</td><td>115</td></tr>
<tr><td>howey/bert-base-uncased-mrpc</td><td>0.850</td><td>0.896</td><td>2.88</td><td>52</td></tr>
<tr><td>NDugar/finetuned-bert-mrpc</td><td>0.846</td><td>0.892</td><td>2.90</td><td>27</td></tr>
<tr><td>shivangi/MRPC_output</td><td>0.831</td><td>0.881</td><td>2.99</td><td>65</td></tr>
<tr><td>M-FAC/bert-mini-finetuned-mrpc</td><td>0.811</td><td>0.865</td><td>1.45</td><td>25</td></tr>
<tr><td>textattack/distilbert-base-cased-MRPC</td><td>0.784</td><td>0.842</td><td>1.01</td><td>122</td></tr>
<tr><td>shivangi/MRPC_64_128_output</td><td>0.772</td><td>0.847</td><td>2.90</td><td>66</td></tr>
<tr><td>ntrnghia/mrpc_vn</td><td>0.738</td><td>0.838</td><td>3.02</td><td>63</td></tr>
<tr><td>M-FAC/bert-tiny-finetuned-mrpc</td><td>0.735</td><td>0.831</td><td>1.29</td><td>36</td></tr>
<tr><td>ji-xin/roberta_base-MRPC-two_stage</td><td>0.686</td><td>0.813</td><td>3.03</td><td>78</td></tr>
<tr><th colspan="5"> BASELINES</th></tr>
<tr><td><a href="https://paperswithcode.com/sota/semantic-textual-similarity-on-mrpc">Papers with code</a></td><td> 0.937 </td><td>  </td><td>  </td><td></td></tr>
<tr><td><a href="https://gluebenchmark.com/leaderboard">GLUE Leaderboard</a></td><td> 0.945 </td><td>  </td><td> 0.926 </td><td></td></tr>
<tr><td><a href="https://gluebenchmark.com/leaderboard">GLUE Human Baseline</a></td><td> 0.863 </td><td> 0.808 </td><td>  </td><td></td></tr>
<tr><td>Best of YuchenBian models</td><td> 0.924 </td><td> 0.946 </td><td>6.38  </td><td></td></tr>


</table>

The last line in the table above is based on: <a href="../../../examples/eval_examples/model.HuggingFace/language/data.HuggingFace/Yuchen_models/glue/mrpc.out">$gft/examples/eval_examples/model.HuggingFace/language/data.HuggingFace/Yuchen_models/glue/mrpc.out</a>.

<p> In general, the most popular models (in terms of downloads)
perform well (in terms of accuracy and f1), though it is often
possible to come up with slightly better models, using methods such as
hyper-parameter tuning.  The top models on leaderboards are likely to
be even better than that.

<p> Many of these models perform well above
human performance.  That might not be as impressive as it seems,
though, since there may well be ways for machine learning to game many
of these tasks.