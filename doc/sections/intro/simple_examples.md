# Some Simple Examples of GFT (General Fine-Tuning)

<h3>Summarize (Almost) Anything: gft_summary</h3>

The following summarizes datasets and/or models:

```sh
emodel=H:bhadresh-savani/roberta-base-emotion
gft_summary --data H:emotion
gft_summary --model $emodel
gft_summary --data H:emotion --model $emodel
```

<h3>How Do I Find the <i>Good</i> Stuff?: __contains__</h3>

We have an embarrassment of riches: 30k models and 3k datasets on hubs such as <a href="https://huggingface.co/">HuggingFace</a>
and <a href="https://github.com/PaddlePaddle/PaddleNLP">PaddleNLP</a>/<a href="https://github.com/PaddlePaddle/PaddleHub">PaddleHub</a>.
How
do we find the <i>good</i> stuff?
<p>
The following finds datasets and models that contain the substring <i>emotion</i> (sorted by downloads):

```sh
gft_summary --data H:__contains__emotion --topn 5
gft_summary --model H:__contains__emotion --topn 5
```

With the empty string, these tools find the topn datasets and models, sorted by downloads:

```sh
gft_summary --data H:__contains__ --topn 50
gft_summary --model H:__contains__ --topn 50
gft_summary --task H:__contains__ 
```

<h3>Inference: gft_predict</h3>

How do I use the <i>good</i> stuff?
Predict takes an input, <i>x</i>, and a model, <i>f</i>,
and returns, <i>f(x)</i>:

```sh
emodel=H:bhadresh-savani/roberta-base-emotion
x='I love you.'
echo $x | gft_predict --task H:classify
echo $x | gft_predict --task H:classify --model $emodel
```

The default model (for the classify task) is sentiment analysis.
With the model, $emodel, the output labels are emotion labels: anger, fear, joy, love, sadness, surprise.

<h3>More Tasks</h3>

```sh
echo 'I love New York.' | gft_predict --task H:token-classification
echo 'I <mask> you.' | gft_predict --task H:fill-mask
```

<h3>Input from Datasets (as opposed to stdin)</h3>

The following reads input from the text column of the test split of the emotion dataset from HuggingFace.
The equation specifies the task and column(s).

```sh
emodel=H:bhadresh-savani/roberta-base-emotion
gft_predict --eqn 'classify: label ~ text' --model $emodel --data H:emotion --split test 2>/dev/null | head
```

<h3>Evaluation: gft_eval</h3>

gft_predict outputs a prediction for each input.  gft_eval reduces these predictions to a single score.

```sh
emodel=H:bhadresh-savani/roberta-base-emotion
gft_eval --eqn 'classify: label ~ text' --model $emodel --data H:emotion --split test 2>/dev/null 
```

<h3>Fine-Tuning: gft_fit</h3>

gft_fit inputs a pre-trained model, and outputs a post-trained model fit to data.
The following assumes that output_dir points to a place with plenty of disk space.

```sh
model=H:best-base-cased
gft_fit --eqn 'classify: label ~ text' --model $model --data H:emotion --output_dir $outdir
```

The following example is similar but uses a model and a dataset from PaddleNLP (as opposed to HuggingFace).
<i>gft</i> is designed to make it easy to mix and match datasets and models from different hubs (suppliers).

```sh
model=P:ernie-tiny
data=P:chnsenticorp
gft_fit --eqn 'classify: label ~ text' --model $model --data $data --output_dir $outdir
```

