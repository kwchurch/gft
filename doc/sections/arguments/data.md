# Data

The following will output the 10 most popular datasets on HuggingFace that contain the substring: <i>emotion</i>.

```sh
gft_summary --data H:__contains__emotion
```


The following will produce a long list of datasets that are available on
(H) <a href="https://huggingface.co/datasets">HuggingFace</a> and (P) <a
href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/datasets">PaddleNLP</a>.

```sh
gft_summary --data H:__contains__ --topn 10000000 --fast_mode
gft_summary --data P:__contains__ --topn 10000000 --fast_mode
```

In addition to these datasets, one can also create models on the local disk with <a href="../functions/gft_fit.md">gft_fit</a> (or any other method).
<p>
Many of <i>gft</i> functions take a --data argument.  The value of this argument is typically: <i>data_provider</i>:<i>data_key</i>, where <i>data_provider</i> is a hub, H or P, and <i>data_key</i> is the name
of a dataset supported by the hub.
<p>
HuggingFace provides a convenient GUI for looking at datasets.  Here are a few popular datasets:
<ol>
<li><a href="https://huggingface.co/datasets/glue">glue</a></li>
<li><a href="https://huggingface.co/datasets/squad">squad</a></li>
<li><a href="https://huggingface.co/datasets/emotion">emotion</a></li>
<li><a href="https://huggingface.co/datasets/common_voice">common_voice</a></li>
</ol>
Datasets typically contain splits such as train, val, test.  Some datasets have more (or less) splits.  Splits go by various names.  Val is often called valid, validation, dev.
<p>

Each split typically contains a number of rows and columns.  The squad
dataset, for example, has 5 columns: id, title, context, question,
answers.  There are typically hundreds or thousands of rows.  Rows are
typically assumed to be <a
href="https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables">iid</a>,
though this assumption is often idealistic (and unrealistic).

<p>

There tend to be more papers on models than datasets.  There is
relatively little interest in what the dataset is testing.  Is the
dataset realistic?  What is the motivation?  Who is the intended
audience?  Lexicographers care about sampling and balance.  Many of
these datasets can be gamed.  That is, there are often (spurious) features in the training split
that make it possible for a system to do well on predicting labels on the test split without
mastering the task.  This kind of leakage between training and set splits is referred to as <i>spurious
correlations</i>.  See <a href="http://proceedings.mlr.press/v119/sagawa20a.html">here</a> for a recent paper on this topic.

<p>
Psychology has a considerble literature on <a href="https://en.wikipedia.org/wiki/Validity_(statistics)">validity</a> and <a href="https://en.wikipedia.org/wiki/Reliability_(statistics)">reliability</a>.
This discussion presupposes the existence of a hypothesis.  Datasets are collected in order
to test that hypothesis.  There tends to be less emphasis on hypothesis testing in machine learning.
