# gft (general fine-tuning): gft_fit

Most <i>gft</i> programs are short (1-liners).  While <i>gft</i> supports most arguments
in most HuggingFace and PaddleNLP examples, most <i>gft</i> programs specify 4 arguments:

<ul>
<li><a href="../arguments/data.md">Data</a></li>
<li><a href="../arguments/model.md">Models</a></li>
<li><a href="../arguments/eqn.md">Equations</a></li>
<li><a href="../argument/task.md">Tasks</a></li>
</ul>


The following example of <i>gft_fit</i> takes a pre-trained model, <a href="https://huggingface.co/bert-base-cased">bert-base-cased (from HuggingFace)</a>, as input,
and outputs a post-trained model to $outdir.  ($outdir should be a place with plenty of disk space, because the output
models are as large as the input pre-trained models.)
<p>
This example uses the <a href="https://huggingface.co/datasets/glue">glue dataset (from HuggingFace)</a> to fine-tune (or fit) the pre-trained model for the qqp task.


```sh
gft_fit --model H:bert-base-cased \
    --data H:glue,qqp \
    --metric H:glue,qqp \
    --output_dir $outdir \
    --eqn 'classify: label ~ question1 + question2'
```

Many examples of <i>gft_fit</i> can be found <a href="../../../examples/fit_examples">here</a>.

One of the design goals of <i>gft</i> is to make it easy to mix and match models and datasets
from different suppliers.  The following uses <a href="https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html">models</a>
and <a href="https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/datasets">data</a> from PaddleNLP (as opposed to HuggingFace).

```sh
gft_fit --model P:bert-base-cased \
    --data P:glue,qqp \
    --metric H:glue,qqp \
    --output_dir $outdir \
    --eqn 'classify: labels ~ sentence1 + sentence2' \
```

The variables in the equations refer to columns in the datasets.  The
equations are slightly different in two gft programs above
because different suppliers of glue data use different names for columns.
<p>
There are 4 examples of the glue tasks (for 4 combinations of suppliers of datasets and models):

<table>
<tr><th>Model</th><th>Data</th><th>Example</th></tr>
<tr><td>H</td><td>H</td><td><a href="../../../examples/fit_examples/model.HuggingFace/language/data.HuggingFace/glue">here</a></td></tr>
<tr><td>H</td><td>P</td><td><a href="../../../examples/fit_examples/model.HuggingFace/language/data.PaddleHub/glue">here</a></td></tr>
<tr><td>P</td><td>H</td><td><a href="../../../examples/fit_examples/model.PaddleHub/language/data.HuggingFace/glue">here</a></td></tr>
<tr><td>P</td><td>P</td><td><a href="../../../examples/fit_examples/model.PaddleHub/language/data.PaddleHub/glue">here</a></td></tr>
</table>

The following will run one of these examples:

```sh
export datasets=$gft/datasets
outdir=/tmp/cola/cpkt
sh $gft/examples/fit_examples/model.HuggingFace/language/data.HuggingFace/glue/cola.sh $outdir
```

<p>
To run all fit examples:

```
cd $gft/examples/fit_examples
find . -name '*.sh' |
while read f
do
b=$checkpoints/`dirname $f`/`basename $f .sh`
sh $f $b/ckpt
done
```

One of the design goals of <i>gft</i> is to make fine-tuning as accessible to
a broad audience as possible.  It should be as easy to fine-tune a
deep net as it is to fit a regression model.

<i>gft</i> equations are similar to <a href="https://en.wikipedia.org/wiki/General_linear_model">glm</a> (general linear models)
equations in regression packages such as in <a href="https://www.r-project.org/">R</a>.




<p>
All of the shell scripts under fine_tuning_examples take a single argument (a directory for the results).

<p>
The shell scripts under model.HuggingFace use models from
HuggingFace, and shell scripts under model.PaddleHub use models from
PaddleHub and/or PaddleNLP.
Similarly, shell scripts under data.HuggingFace use datasets from
HuggingFace, and shell scripts under data.PaddleHub use datasets from
PaddleHub and/or PaddleNLP.

