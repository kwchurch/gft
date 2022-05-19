# gft (general fine-tuning): A Little Language for Deepnets

1-line programs for fine-tuning, inference and more


<h1>Quick Links</h1>
<ol>
<li>Videos 📽️ <ol>
	<li>📽️ <a href="https://youtu.be/IKjx38AV4bo">10 minute TEASER</a></li>
	<li>🆕📽️ <a href="https://youtu.be/fWS-pgKYE_o">First half of ACL-2022 Tutorial (1 hour 16 minutes) UNABRIDGED</a></li>
	</ol></li>
<li><a href="doc/sections/installation.md">Installation</a></li>
<li><a href="doc/README.md">Documentation</a></li>
<li><a href="https://github.com/kwchurch/ACL2022_deepnets_tutorial">ACL-2022 Tutorial</a></li>
</ol>

<h1>Four Functions and Four Arguments</h1>

<i>gft</i> contains 4 main functions:
<ol>
<li><a href="doc/sections/functions/gft_fit.md">gft_fit</a>: fit a pretrained model to data (aka <i>fine-tuning</i>)</li>
<li><a href="doc/sections/functions/gft_predict.md">gft_predict</a>: apply a model to inputs (aka <i>inference</i>)</li>
<li><a href="doc/sections/functions/gft_eval.md">gft_eval</a>: score a model on a split of a dataset</li>
<li><a href="doc/sections/functions/gft_summary.md">gft_summary</a>: Find good stuff (popular models and datasets), and explain what's in those models and datasets.</li>
</ol>

These <i>gft</i> functions make use of 4 main arguments (though most arguments in most hubs are also supported):

<ol>
<li><a href="doc/sections/arguments/data.md">data</a>: standard datasets hosted on hubs such as HuggingFace, PaddleNLP, or custom datasets hosted on the local filesystem </li>
<li><a href="doc/sections/arguments/model.md">model</a>: standard models hosted on hubs such as HuggingFace, PaddleNLP, or custom models hosted on the local filesystem </li>
<li><a href="doc/sections/arguments/eqn.md">equation</a>: string such as "classify: label ~ text", where classify is a task, and label and text refer to columns in a dataset </li>
<li><a href="doc/sections/arguments/task.md">task</a>: classify, classify_tokens, classify_spans, classify_audio, classify_images, regress, text-generation, translation, ASR, fill-mask</li>
</ol>

<h1>A Few Simple Examples</h1>

Here are some simple examples:

```sh
emodel=H:bhadresh-savani/roberta-base-emotion

# Summarize a dataset and/or model
gft_summary --data H:emotion
gft_summary --model $emodel
gft_summary --data H:emotion --model $emodel

# find some popular datasets and models that contain "emotion"
gft_summary --data H:__contains__emotion --topn 5
gft_summary --model H:__contains__emotion --topn 5

# make predictions on inputs from stdin
echo 'I love you.' | gft_predict --task classify

# The default model (for the classification task) performs sentiment analysis
# The model, $emodel, outputs emotion classes (as opposed to POSITIVE/NEGATIVE)
echo 'I love you.' | gft_predict --task classify --model $emodel

# some other tasks (beyond classification)
echo 'I love New York.' | gft_predict --task H:token-classification
echo 'I <mask> you.' | gft_predict --task H:fill-mask

# make predictions on inputs from a split of a standard dataset
gft_predict --eqn 'classify: label ~ text' --model $emodel --data H:emotion --split test

# return a single score (as opposed to a prediction for each input)
gft_eval --eqn 'classify: label ~ text' --model $emodel --data H:emotion --split test

# Input a pre-trained model (bert) and output a post-trained model
gft_fit --eqn 'classify: label ~ text' \
	--model H:bert-base-cased \
	--data H:emotion \
	--output_dir $outdir
```

<h1> Pre-Training, Fine-Tuning and Inference </h1>

The table below shows a 3-step recipe, which has become standard in the literature on deep nets.


<table>
<tr> <th> <b>Step</b> </th> 
     <th> <b>gft Support</b> </th> 
     <th> <b>Description</b> </th>
     <th> <b>Time</b>  </th>
     <th> <b>Hardware</b> </th> 
</tr>

<tr> <td> 1 </td> <td> </td> <td> Pre-Training </td> <td> Days/Weeks </td> <td> Large GPU Cluster </td> </tr>
<tr> <td> 2 </td> <td> <i>gft_fit</i> </td> <td> Fine-Tuning </td> <td> Hours/Days </td> <td> 1+ GPUs </td> </tr>
<tr> <td> 3 </td> <td> <i>gft_predict</i> </td> <td> Inference </td> <td> Seconds/Minutes </td> <td> 0+ GPUs </td> </tr>
</table>

This repo provides support for step 2 (<i>gft_fit</i>) and step 3
(<i>gft_predict</i>).  Most <i>gft_fit</i> and <i>gft_predict</i>
programs are short (1-line), much shorter than examples such as <a
href="https://github.com/huggingface/transformers/tree/master/examples">these</a>,
which are typically a few hundred lines of python.  With <i>gft</i>, users should not need to read or modify any python
code for steps 2 and 3 in the table above.

Step 1, pre-training, is beyond the scope of this work.  We recommend
starting with models from HuggingFace and PaddleHub/PaddleNLP hubs, as
illustrated in the examples below.

<h1>Citations, Documentation, etc.</h1>

Paper is <a href="https://aclanthology.org/2022.acl-tutorials.1/">here.</a>

```sh
@inproceedings{church-etal-2022-gentle,
    title = "A Gentle Introduction to Deep Nets and Opportunities for the Future",
    author = "Church, Kenneth  and
      Kordoni, Valia  and
      Marcus, Gary  and
      Davis, Ernest  and
      Ma, Yanjun  and
      Chen, Zeyu",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: Tutorial Abstracts",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-tutorials.1",
    pages = "1--6",
    abstract = "The first half of this tutorial will make deep nets more accessible to a broader audience, following {``}Deep Nets for Poets{''} and {``}A Gentle Introduction to Fine-Tuning.{''} We will also introduce GFT (general fine tuning), a little language for fine tuning deep nets with short (one line) programs that are as easy to code as regression in statistics packages such as R using glm (general linear models). Based on the success of these methods on a number of benchmarks, one might come away with the impression that deep nets are all we need. However, we believe the glass is half-full: while there is much that can be done with deep nets, there is always more to do. The second half of this tutorial will discuss some of these opportunities.",
}```
