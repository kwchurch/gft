# gft (general fine-tuning): A Little Language for Deepnets

1-line programs for fine-tuning and inference

<h1>Installation</h1>

Clone this repo and then say:

```sh
pip install -r requirements/requirements.txt
```

Make sure <a href="gft_predict">gft_predict</a> is in your PATH.  Then set up the following environment variables with:

```sh
g=`which gft_predict`
export gft=`dirname $g`
```

<b style="background-color:yellow;">Unfortunately</b>, there are a number of incompatibilities between adapters, paddlespeech and
the latest version of HuggingFace transformers. There are a number of versions of requirements.txt the requirements directory.
We recommend setting up several different virtual environments to work around some of these incompatibilities.


The scripts in the examples directory will create results under
<i>$gft_checkpoints</i>.  Please set that variable to some place where
you have plenty of free disk space.  The results are large because
most fine-tuning examples copy a pre-trained model.  Given there are
many dozens of such examples, there will be many dozens of copies of
large models.

<p>
WARNING: Some of the fine-tuning scripts take a long time, and not all examples are working (yet).

<h1> Introduction: Pre-Training, Fine-Tuning and Inference </h1>

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

<h2>Fine-Tuning Equations</h2>

Most <i>gft</i> programs are short (1-liners).  While <i>gft</i> supports most arguments
in most HuggingFace and PaddleNLP examples, most <i>gft</i> programs specify 4 arguments:

```sh
gft_fit --model H:bert-base-cased \
    --data H:glue,qqp \
    --metric H:glue,qqp \
    --output_dir $outdir \
    --eqn 'classify: label ~ question1 + question2'
```

This gft program fine-tunes a pretrained model (bert-base-cased from
HuggingFace) with a dataset (the qqp subset of glue from HuggingFace).

One of the design goals of <i>gft</i> is to make fine-tuning as accessible to
a broad audience as possible.  It should be as easy to fine-tune a
deep net as it is to fit a regression model.

<i>gft</i> equations are similar to <a href="https://en.wikipedia.org/wiki/General_linear_model">glm</a> (general linear models)
equations in regression packages such as in <a href="https://www.r-project.org/">R</a>.

Another design goal is to make it easy to port from one supplier to another.
The <i>gft</i> program below is similar to the one above, except this one
uses data from PaddleNLP instead of HuggingFace.  With <i>gft</i>, it should
be possible to mix and match datasets and models from different suppliers.

The variables in the equations refer to columns in the datasets.  The
equations are slightly different in gft programs above and below
because different suppliers use different names for columns.

```sh
gft_fit --model H:bert-base-cased \
    --data P:glue,qqp \
    --metric H:glue,qqp \
    --output_dir $outdir \
    --eqn 'classify: labels ~ sentence1 + sentence2'
```

<h2>Datasets and Models: Embarrassement of Riches</h2>

As mentioned above, there are about 30k models and 3k datasets in
<a href="https://huggingface.co/">HuggingFace</a>,
and there are more in 
<a href="https://www.paddlepaddle.org.cn/hublist">PaddleHub/PaddleNLP</a>
and <a href="https://adapterhub.ml/">AdapterHub</a>.  Going forward, there
will be even more.  These numbers are about 3x larger than they were a
year ago.  How can we find the good stuff?


Some datasets are more popular than others (<a href="huggingface_hub/best_of/datasets/">top 1000</a>):

<img src="huggingface_hub/downloads.jpg" alt="downloads" width=700>

Top 50 HuggingFace datasets by two metrics (both if member of both sets):
<ul>
<li> <b># of models</b>: 
<b><a href="https://huggingface.co/models?dataset=dataset:glue">glue</a></b>,
<a href="https://huggingface.co/models?dataset=dataset:super_glue">super_glue</a>,
<b><a href="https://huggingface.co/models?dataset=dataset:common_voice">common_voice</a></b>,
<b><a href="https://huggingface.co/models?dataset=dataset:librispeech_asr">librispeech_asr</a></b>,
<a href="https://huggingface.co/models?dataset=dataset:wikitext">wikitext</a>,
<b><a href="https://huggingface.co/models?dataset=dataset:squad">squad</a></b>,
<b><a href="https://huggingface.co/models?dataset=dataset:imdb">imdb</a></b>,
<a href="https://huggingface.co/models?dataset=dataset:blimp">blimp</a>,
<a href="https://huggingface.co/models?dataset=dataset:paws">paws</a>,
<b><a href="https://huggingface.co/models?dataset=dataset:wmt16">wmt16</a></b>,
<b><a href="https://huggingface.co/models?dataset=dataset:tweet_eval">tweet_eval</a></b>,
<a href="https://huggingface.co/models?dataset=dataset:trec">trec</a>,
<a href="https://huggingface.co/models?dataset=dataset:red_caps">red_caps</a>,
<b><a href="https://huggingface.co/models?dataset=dataset:xnli">xnli</a></b>,
<a href="https://huggingface.co/models?dataset=dataset:ag_news">ag_news</a>,
<a href="https://huggingface.co/models?dataset=dataset:tweets_hate_speech_detection">tweets_hate_speech_detection</a>,
<b><a href="https://huggingface.co/models?dataset=dataset:squad_v2">squad_v2</a></b>,
<a href="https://huggingface.co/models?dataset=dataset:stsb_multi_mt">stsb_multi_mt</a>,
<a href="https://huggingface.co/models?dataset=dataset:cfq">cfq</a>,
<b><a href="https://huggingface.co/models?dataset=dataset:wikiann">wikiann</a></b>,
<a href="https://huggingface.co/models?dataset=dataset:anli">anli</a>,
<b><a href="https://huggingface.co/models?dataset=dataset:conll2003">conll2003</a></b>,
<b><a href="https://huggingface.co/models?dataset=dataset:snli">snli</a></b>,
<a href="https://huggingface.co/models?dataset=dataset:schema_guided_dstc8">schema_guided_dstc8</a>,
<a href="https://huggingface.co/models?dataset=dataset:hellaswag">hellaswag</a>,
<b><a href="https://huggingface.co/models?dataset=dataset:xsum">xsum</a></b>,
<b><a href="https://huggingface.co/models?dataset=dataset:code_search_net">code_search_net</a></b>,
<a href="https://huggingface.co/models?dataset=dataset:winogrande">winogrande</a>,
<b><a href="https://huggingface.co/models?dataset=dataset:cnn_dailymail">cnn_dailymail</a></b>,
<a href="https://huggingface.co/models?dataset=dataset:tab_fact">tab_fact</a>,
<a href="https://huggingface.co/models?dataset=dataset:piqa">piqa</a>,
<a href="https://huggingface.co/models?dataset=dataset:adversarial_qa">adversarial_qa</a>,
<a href="https://huggingface.co/models?dataset=dataset:banking77">banking77</a>,
<b><a href="https://huggingface.co/models?dataset=dataset:race">race</a></b>,
<a href="https://huggingface.co/models?dataset=dataset:amazon_polarity">amazon_polarity</a>,
<b><a href="https://huggingface.co/models?dataset=dataset:emotion">emotion</a></b>,
<a href="https://huggingface.co/models?dataset=dataset:americas_nli">americas_nli</a>,
<a href="https://huggingface.co/models?dataset=dataset:hans">hans</a>,
<a href="https://huggingface.co/models?dataset=dataset:daily_dialog">daily_dialog</a>,
<a href="https://huggingface.co/models?dataset=dataset:wino_bias">wino_bias</a>,
<b><a href="https://huggingface.co/models?dataset=dataset:oscar">oscar</a></b>,
<b><a href="https://huggingface.co/models?dataset=dataset:amazon_reviews_multi">amazon_reviews_multi</a></b>,
<b><a href="https://huggingface.co/models?dataset=dataset:klue">klue</a></b>,
<b><a href="https://huggingface.co/models?dataset=dataset:mc4">mc4</a></b>,
<b><a href="https://huggingface.co/models?dataset=dataset:universal_dependencies">universal_dependencies</a></b>,
<a href="https://huggingface.co/models?dataset=dataset:rotten_tomatoes">rotten_tomatoes</a>,
<b><a href="https://huggingface.co/models?dataset=dataset:samsum">samsum</a></b>,
<a href="https://huggingface.co/models?dataset=dataset:cosmos_qa">cosmos_qa</a>,
<b><a href="https://huggingface.co/models?dataset=dataset:c4">c4</a></b>,
<b><a href="https://huggingface.co/models?dataset=dataset:xtreme">xtreme</a></b> </li>


<li> <b># of downloads</b>: 
<a href="https://huggingface.co/datasets/wikipedia">wikipedia</a>,
<b><a href="https://huggingface.co/datasets/common_voice">common_voice</a></b>,
<a href="https://huggingface.co/datasets/bookcorpus">bookcorpus</a>,
<b><a href="https://huggingface.co/datasets/c4">c4</a></b>,
<b><a href="https://huggingface.co/datasets/glue">glue</a></b>,
<b><a href="https://huggingface.co/datasets/squad">squad</a></b>,
<b><a href="https://huggingface.co/datasets/conll2003">conll2003</a></b>,
<b><a href="https://huggingface.co/datasets/oscar">oscar</a></b>,
<b><a href="https://huggingface.co/datasets/librispeech_asr">librispeech_asr</a></b>,
<b><a href="https://huggingface.co/datasets/tweet_eval">tweet_eval</a></b>,
<b><a href="https://huggingface.co/datasets/squad_v2">squad_v2</a></b>,
<b><a href="https://huggingface.co/datasets/emotion">emotion</a></b>,
<a href="https://huggingface.co/datasets/masakhaner">masakhaner</a>,
<b><a href="https://huggingface.co/datasets/cnn_dailymail">cnn_dailymail</a></b>,
<b><a href="https://huggingface.co/datasets/amazon_reviews_multi">amazon_reviews_multi</a></b>,
<b><a href="https://huggingface.co/datasets/imdb">imdb</a></b>,
<b><a href="https://huggingface.co/datasets/wmt16">wmt16</a></b>,
<b><a href="https://huggingface.co/datasets/mc4">mc4</a></b>,
<b><a href="https://huggingface.co/datasets/xsum">xsum</a></b>,
<a href="https://huggingface.co/datasets/superb">superb</a>,
<b><a href="https://huggingface.co/datasets/universal_dependencies">universal_dependencies</a></b>,
<a href="https://huggingface.co/datasets/multi_nli">multi_nli</a>,
<b><a href="https://huggingface.co/datasets/xtreme">xtreme</a></b>,
<a href="https://huggingface.co/datasets/vctk">vctk</a>,
<a href="https://huggingface.co/datasets/covost2">covost2</a>,
<b><a href="https://huggingface.co/datasets/wikiann">wikiann</a></b>,
<b><a href="https://huggingface.co/datasets/samsum">samsum</a></b>,
<a href="https://huggingface.co/datasets/openslr">openslr</a>,
<a href="https://huggingface.co/datasets/timit_asr">timit_asr</a>,
<b><a href="https://huggingface.co/datasets/xnli">xnli</a></b>,
<b><a href="https://huggingface.co/datasets/snli">snli</a></b>,
<a href="https://huggingface.co/datasets/multilingual_librispeech">multilingual_librispeech</a>,
<a href="https://huggingface.co/datasets/wmt14">wmt14</a>,
<b><a href="https://huggingface.co/datasets/klue">klue</a></b>,
<a href="https://huggingface.co/datasets/gigaword">gigaword</a>,
<a href="https://huggingface.co/datasets/clinc_oos">clinc_oos</a>,
<a href="https://huggingface.co/datasets/cc100">cc100</a>,
<a href="https://huggingface.co/datasets/wmt19">wmt19</a>,
<a href="https://huggingface.co/datasets/natural_questions">natural_questions</a>,
<a href="https://huggingface.co/datasets/nli_tr">nli_tr</a>,
<a href="https://huggingface.co/datasets/mlsum">mlsum</a>,
<b><a href="https://huggingface.co/datasets/code_search_net">code_search_net</a></b>,
<a href="https://huggingface.co/datasets/bigscience/P3">bigscience/P3</a>,
<a href="https://huggingface.co/datasets/kde4">kde4</a>,
<a href="https://huggingface.co/datasets/xquad">xquad</a>,
<a href="https://huggingface.co/datasets/wikisql">wikisql</a>,
<b><a href="https://huggingface.co/datasets/race">race</a></b>,
<a href="https://huggingface.co/datasets/go_emotions">go_emotions</a>,
<a href="https://huggingface.co/datasets/conll2002">conll2002</a>,
<a href="https://huggingface.co/datasets/wnut_17">wnut_17</a></li>

</ul>

HuggingFace provides tools to make it easier to work with large
numbers of models and datasets.  There are some short python programs
in $gft/huggingface_hub that use these tools to list models and
datasets (and useful cominbations of the two).  There are also some
text files that were created by these python programs:

<table>
<tr> <th> N </th>  <th> python program </th> <th> output </th> </tr>
<tr> <td>  29915</td> 
     <td> <a href="huggingface_hub/list_huggingface_models.py">$gft/huggingface_hub/list_huggingface_models.py</a></td> 
     <td> <a href="huggingface_hub/huggingface_models.txt">$gft/huggingface_hub/huggingface_models.txt</a></td></tr>
<tr> <td>     400 </td> 
     <td> <a href="huggingface_hub/list_adapters.py>$gft/huggingface_hub/list_adapters.py</a></td> 
     <td> <a href="huggingface_hub/adapters.txt>$gft/huggingface_hub/adapters.txt</a></td></tr>
<tr> <td>    3098</td>  
     <td> <a href="huggingface_hub/list_huggingface_datasets.py">$gft/huggingface_hub/list_huggingface_datasets.py</a></td>
     <td> <a href="huggingface_hub/huggingface_datasets.txt">$gft/huggingface_hub/huggingface_datasets.txt</a></td></tr>
</table>

There is also a python program, models_for_dataset.py, that outputs a
list of models for a particular dataset.  On average, there are about 10x models for each dataset,
though obviously, some datasets are more popular than others.  Here are a couple of models for a few of the more
popular datasets.


```sh
python $gft/huggingface_hub/models_for_dataset.py common_voice | sed 3q
# dataset: common_voice --> 524 models
common_voice	facebook/wav2vec2-large-xlsr-53
common_voice	facebook/wav2vec2-xls-r-300m

python $gft/huggingface_hub/models_for_dataset.py glue | sed 3q
# dataset: glue --> 188 models
glue	Alireza1044/albert-base-v2-sst2
glue	DeepPavlov/xlm-roberta-large-en-ru-mnli

python $gft/huggingface_hub/models_for_dataset.py emotion | sed 3q
# dataset: emotion --> 59 models
emotion	bhadresh-savani/distilbert-base-uncased-emotion
emotion	nateraw/bert-base-uncased-emotion
```

Here are some possibly useful links for finding datasets and models.

<table>
<tr> <th> </th> <th> HuggingFace </th> <th> PaddleHub/PaddleNLP </th> <th> <a href="https://adapterhub.ml/">AdapterHub</a> </th></tr>
<tr> <th> Datasets </th> 
     <td> <a href="huggingface_hub/huggingface_datasets.txt"> text file </a> 
          <a href="https://huggingface.co/dataset">explorer</a>
	  </td>
     <td><a href="https://paddlenlp.readthedocs.io/en/latest/data_prepare/dataset_list.html"> datasets </a> </td> 
</tr>

<tr> <th> Models </th> 
     <td> <a href="huggingface_hub/huggingface_models.txt"> large text file </a>
 <a href="https://huggingface.co/models">explorer</a>	    
 </td>
     <td> <a href="https://github.com/PaddlePaddle/models"> models </a> </td> 
<td> <a href="huggingface_hub/adapters.txt"> small text file </a> 
 <a href="https://adapterhub.ml/explore/">explorer</a>	    
</td
</tr>
</table>



<h2>gft_dataset: Output Dataset on stdout</h2>

The program, gft_dataset, makes it easy to look at datasets from
different suppliers.  The following example downloads the qqp subset
of glue from HuggingFace and PaddleNLP.  The "2>/dev/null" removes
messages sent to stderr.  Piping the results to "sed 1q" terminates
output after the first line (for expository convenience).  Note that
the HuggingFace version (H:) names the fields: question1, question2,
label, and the PaddleNLP version (P:) names the fields: sentence1,
sentence2, labels.

```sh
# Go to Huggingface, and output the val split of the qqp task in glue to stdout
gft_dataset --data 'H:glue,qqp' --split val 2>/dev/null | sed 1q
question1|Why are African-Americans so beautiful?	question2|Why are hispanics so beautiful?	label|0	idx|0

# Same as above, but replace HuggingFace with PaddleNLP
gft_dataset --data 'P:glue,qqp' --split val 2>/dev/null | sed 1q
sentence1|Why are African-Americans so beautiful?	sentence2|Why are hispanics so beautiful?	labels|0
```

If the optional eqn argument is provided, then <i>gft_dataset</i> uses the equation to extract the appropriate fields.
The first column of the output corresponds to the rhs (right hand side) of the equation,
and the second column corresponds to the lhs (left hand side) of the equation.

```sh
# Same as above, but use the equation to select fields of interest
gft_dataset --eqn 'classify: label ~ question1 + question2' --data 'H:glue,qqp' --split val 2>/dev/null | sed 1q
Why are African-Americans so beautiful?|Why are hispanics so beautiful?	0
```

The following example illustrates a custom dataset (C:), where the data are in csv files on the load filesystem.
Normally, the lhs of a regression is a single real value,
but in this case, it is a vector in R^3 .

```sh
# without eqn arg
gft_dataset --data "C:$gft/datasets/VAD/VAD" --split val 2>/dev/null | sed 3q
# Word|abandonment	Valence|0.128	Arousal|0.43	Dominance|0.202
# Word|abbey	Valence|0.58	Arousal|0.367	Dominance|0.444
# Word|abbreviation	Valence|0.469	Arousal|0.306	Dominance|0.345

# with eqn arg
gft_dataset --eqn 'regress: Valence + Arousal + Dominance ~ Word' --data "C:$gft/datasets/VAD/VAD" --split val 2>/dev/null | sed 3q
# abandonment	0.128|0.43|0.202
# abbey	0.58|0.367|0.444
# abbreviation	0.469|0.306|0.345
```

The following example shows that <i>gft_dataset</i> can also be applied to speech datasets.  Common voice is available in English (en),
Chinese (zh-CN), as well as a number of other choices.  The raw data includes the waveform as an array, but with the eqn argument,
we can extract a few useful fields such as the filename and the transcription.

```sh
# without eqn arg
gft_dataset --data H:common_voice,en 2>/dev/null | sed 1q | tr '\t' '\n' 
client_id|a07b17f8234ded5e847443ea6f423cef745cbbc7537fb637d58326000aa751e829a21c4fd0a35fc17fb833aa7e95ebafce5e...
# path|common_voice_en_100363.mp3
# audio|{'path': 'cv-corpus-6.1-2020-12-11/en/clips/common_voice_en_100363.mp3', 'array': array([0.0000000e+...
# sentence|It was the time of day when all of Spain slept during the summer.
# up_votes|2
# down_votes|1
# age|
# gender|
# accent|
# locale|en
# segment|''

# with eqn arg (English)
gft_dataset --eqn 'ctc:sentence ~ path'  --data H:common_voice,en 2>/dev/null | sed 3q
# common_voice_en_100363.mp3	It was the time of day when all of Spain slept during the summer.
# common_voice_en_100540.mp3	Same way you did.
# common_voice_en_100546.mp3	Sarah told him that she was there to see her brother.

# with eqn arg (Chinese)
gft_dataset --eqn 'ctc:sentence ~ path'  --data H:common_voice,zh-CN 2>/dev/null | sed 3q
# common_voice_zh-CN_18524189.mp3	正巧母亲往外探头
# common_voice_zh-CN_18532640.mp3	至今为止，元气火箭总共发行了两张专辑。
# common_voice_zh-CN_18532644.mp3	失业率降到十七年来的新低点
```

<h2>Labels</h2>

<i>gft_labels.py</i> outputs the set of labels for datasets and/or models.

```sh
./gft_labels.py --data H:emotion 2>/dev/null
H:emotion	sadness	joy	love	anger	fear	surprise
```

```sh
./gft_labels.py --model H:AdapterHub/bert-base-uncased-pf-emotion 2>/dev/null
H:AdapterHub/bert-base-uncased-pf-emotion	sadness	joy	love	anger	fear	surprise
```

```sh
./gft_labels.py --task image-classification --model H:nateraw/vit-base-cats-vs-dogs 2>/dev/null
H:nateraw/vit-base-cats-vs-dogs	cat	dog
```

```sh
# The default model for text-classification has 2 classes
gft_labels.py --model H:distilbert-base-uncased-finetuned-sst-2-english
# H:distilbert-base-uncased-finetuned-sst-2-english	NEGATIVE	POSITIVE
```


```sh
# The default model for image-classification has 1k classes
gft_labels.py --model H:google/vit-base-patch16-224 --task image-classification | 
tr '\t' '\n' | wc -l
#   1000

gft_labels.py --model H:google/vit-base-patch16-224 --task image-classification | tr '\t' '\n' | head
# Afghan hound, Afghan
# African chameleon, Chamaeleo chamaeleon
# African crocodile, Nile crocodile, Crocodylus niloticus
# African elephant, Loxodonta africana
# African grey, African gray, Psittacus erithacus
# African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus
# Airedale, Airedale terrier
# American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier
# American alligator, Alligator mississipiensis
```

Note, <i>gft_labels.py</i> uses a set of heuristics.  Sometimes, these heuristics fail to find the names of the labels.

See <a href="huggingface_hub/huggingface_models_with_labels.txt">$gft/huggingface_hub/huggingface_models_with_labels.txt</a>
for labels for about 1000 text-classifier models.

<h2>gft Equations</h2>

The examples above (and below) make use of equations such as:

```sh
--eqn 'classify: label ~ question1 + question2'
```

The keyword, classify, will be distinguished from other keywords such
as regress, classify_tokens, classify_spans, ctc, etc.  In the
classification case, for each input (pairs of two questions), there is
a single label (semantically similar or not).  Classify also
generalizes from binary classification to multiclass classification
(for tasks such as emotion classification).  As shown in the table below,
equations start with a number of different keywords:
<ol>
<li> <i>classify</i>: lhs denotes a set of classes </li>
<li> <i>regress</i> : lhs denotes a point in a vector space </li>
<li> <i>classify_tokens</i> : there is a classification task for each token on the rhs</li>
<li> <i>classify_spans</i> : used for SQuAD-like tasks where the output should be a span (substring) of the rhs </li>
<li> <i>ctc</i>: used in speech recognition where the input is audio and the output is text </li>
</ol>

There are a number of examples of equations in the table below.

<table>
<tr> <th><b>Dataset</b></th> <th> <b>Subset</b> </th> <th> <b>Data Argument</b> </th> <th><b>Equation</B></th> <th><b><a href="https://huggingface.co/docs/transformers/main_classes/pipelines">Pipeline Task</a></b></th></tr>
<tr> <td> GLUE </td> <td> COLA </td> <td> H:glue,cola </td> <td> classify : label ~ sentence</td> <td> text-classification <td> </tr>
<tr> <td> GLUE </td> <td> SST2 </td> <td> H:glue,cola </td> <td> classify : label ~ sentence</td> <td> text-classification <td></tr>
<tr> <td> GLUE </td> <td> WNLI </td> <td> H:glue,cola </td> <td> classify : label ~ sentence</td> <td> text-classification <td> </tr>
<tr> <td> GLUE </td> <td> MRPC </td> <td> H:glue,cola </td> <td> classify : label ~ sentence1 + sentence2</td> <td> text-classification <td></tr>
<tr> <td> GLUE </td> <td> QNLI </td> <td> H:glue,cola </td> <td> classify : label ~ sentence1 + sentence2</td> <td> text-classification <td></tr>
<tr> <td> GLUE </td> <td> QQP </td> <td> H:glue,cola </td> <td> classify : label ~ question + sentence</td> <td> text-classification <td></tr>
<tr> <td> GLUE </td> <td> SSTB </td> <td> H:glue,cola </td> <td> regress : label ~ question1 + question2</td> <td>  <td></tr>
<tr> <td> GLUE </td> <td> MNLI </td> <td> H:glue,cola </td> <td> classify : label ~ premise + hypothesis</td> <td> text-classification <td></tr>

<tr> <td> SQuAD 1.0 </td> <td>  </td> <td> H:squad </td> <td> classify_spans : answers ~ question + context </td> <td> question-answering <td></tr>
<tr> <td> SQuAD 2.0 </td> <td>  </td> <td> H:squad_v2 </td> <td> classify_spans : answers ~ question + context </td> <td> question-answering <td></tr>

<tr> <td> CONLL2003 </td> <td>  POS  </td> <td> H:conll2003 </td> <td> classify_tokens : pos_tags ~ tokens </td> <td> token-classification <td></tr>
<tr> <td> CONLL2003 </td> <td>  NER  </td> <td> H:conll2003 </td> <td> classify_tokens : ner_tags ~ tokens </td> <td> token-classification <td></tr>
<tr> <td> CONLL2003 </td> <td>  Chunking  </td> <td> H:conll2003 </td> <td> classify_tokens : chunk_tags ~ tokens </td> <td> token-classification <td></tr>

<tr> <td> TIMIT</td> <td>  </td> <td> H:timit_asr </td> <td> ctc: text ~ audio </td> <td> automatic-speech-recognition<td></tr>
<tr> <td> LibriSpeech</td> <td>  </td> <td> H:librispeech_asr </td> <td> ctc: text ~ audio </td> <td> automatic-speech-recognition<td></tr>
<tr> <td> Amazon Reviews</td> <td>  </td> <td> H:amazon_reviews_multi </td> <td> classify: stars ~ review_title + review_body</td> <td>text-classification<td></tr>
<tr> <td> VAD </td> <td>  </td> <td> C:$gft/datasets/VAD/VAD </td> <td> regress: Valence + Arousal + Dominance ~ Word </td> <td> <td></tr>
</table>

<h1>Inference: gft_predict</h1>

<i>gft_predict</i> reads input from stdin,
and writes output predicts to stdout.
If one wants to run a model on data from a standard dataset,
then pipe <i>gft_dataset</i> into <i>gft_predict</i>.

```sh
# extract examples from dataset (and stop after 2nd one, for expository purposes)
gft_dataset --data H:emotion --eqn 'classify: label ~ text' 2>/dev/null | sed 2q
# im feeling rather rotten so im not very ambitious right now	0
# im updating my blog because i feel shitty	0

# same as above, but pipe results into gft_predict
# gft_predict appends predicted labels and scores to each input line
gft_dataset --data H:emotion --eqn 'classify: label ~ text' 2>/dev/null | sed 2q | 
gft_predict --task H:text-classification 2>/dev/null
# im feeling rather rotten so im not very ambitious right now	0	NEGATIVE	0.9998108744621277
# im updating my blog because i feel shitty	0	NEGATIVE	0.9994602799415588
```

<i>gft_predict</i> will be discussed in more detail <a href="#inference">below</a>.  <i>gft_predict</i> reads from stdin and applies almost any input to almost any model.
<i>gft_predict</i> supports most of the tasks in <a href="https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/pipelines">HuggingFace pipelines</a>.

Here are some examples of inference (see <a href="#inference">for more details</a>):

```sh
# text-classification: sentiment analysis
echo 'I love you.' | gft_predict --task H:text-classification
# I love you.	POSITIVE	0.9998705387115479

# text-classification: emotion classification
model=H:AdapterHub/bert-base-uncased-pf-emotion
echo 'I love you.' | gft_predict --model $model --task H:text-classification
# I love you.	love	0.6005669236183167

# token-classification: NER (Named Entity Recognition)
echo 'I love New York.' | gft_predict --task H:token-classification
# I love New York.	New/I-LOC:0.9989	York/I-LOC:0.9974

# fill-mask: guess the masked word
echo 'I <mask> you.' | gft_predict --task H:fill-mask
# I <mask> you.	 salute|0.241	 miss|0.177	 love|0.147	 thank|0.060	 applaud|0.047

# text-generation 
echo 'I love ' | gft_predict --task H:text-generation 
# I love you and I will never be forgotten and thank you." I was also
# inspired by all of the students who walked onto campus wearing these
#  teddy I love the idea that you can be anything people ask for you

# translation
echo 'I love you.' | gft_predict --task H:translation --model H:Helsinki-NLP/opus-mt-en-fr
# I love you.	Je t'aime.
echo 'I love you.' | gft_predict --task H:translation --model H:Helsinki-NLP/opus-mt-en-zh
I love you.	我爱你
```

```sh
# Run a half-dozen fake-news classifiers on  "I love you."
lab=$gft/huggingface_hub/huggingface_models_with_labels.txt
for model in `egrep fake $lab | cut -f1`
     do
     out=`echo 'I love you.' | gft_predict --task H:text-classification --model $model 2>/dev/null`
     echo $model $out
     done
# elozano/bert-base-cased-fake-news I love you. Fake 0.9996728897094727
# Narrativaai/fake-news-detection-spanish I love you. FAKE 0.9591125845909119
# dtam/autonlp-covid-fake-news-36839110 I love you. 1 0.999913215637207
# Qinghui/autonlp-fake-covid-news-36769078 I love you. 1 0.9999946355819702
# Qiaozhen/fake-news-detector I love you. fake 0.9575745463371277
# yaoyinnan/roberta-fakeddit I love you. Fake 0.9857746362686157
```

```sh
# Run a half-dozen sentiment classifiers on  "I love you."
lab=$gft/huggingface_hub/huggingface_models_with_labels.txt
# Since there are so many (196) classifiers, take a random sample of 6
egrep -ci positive $lab
# 196
models=`awk 'NF < 5' $lab | egrep -i positive  | 
	awk '{print rand() "\t" $0}' | sort | cut -f2- | sed 6q | cut -f1`
for model in $models
     do
     out=`echo 'I love you.' | gft_predict --task H:text-classification --model $model 2>/dev/null`
     echo $model $out
     done
# rohansingh/autonlp-Fake-news-detection-system-29906863 I love you. positive 0.6512816548347473
# gchhablani/fnet-base-finetuned-sst2 I love you. positive 0.9974162578582764
# cointegrated/rubert-tiny-sentiment-balanced I love you. positive 0.9445993304252625
# SetFit/deberta-v3-large__sst2__train-8-5 I love you. positive 0.8264811635017395
# bowipawan/bert-sentimental I love you. positive 0.7457774877548218
# m3tafl0ps/autonlp-NLPIsFun-251844 I love you. positive 0.9010641574859619
```

<table>
<tr><td colspan="4" style="background-color:yellow;"> <i>I love you</i> is <b>positive<b></td><tr>
<tr> <th> <b>Predicted Label</b></th> <th><b>Score</b></th> <th><b>Model</b> </th> <th> <b>Labels for Model</b></th></tr>
<tr><td>positive</td> <td>0.512</td> <td>SetFit/deberta-v3-large__sst2__train-16-7</td><td>negative, positive</td></tr>
<tr><td>POSITIVE</td> <td>0.871</td> <td>ayameRushia/roberta-base-indonesian-sentiment-analysis-smsa</td><td>POSITIVE, NEUTRAL, NEGATIVE</td></tr>
<tr><td>positive</td> <td>0.807</td> <td>SetFit/distilbert-base-uncased__sst2__train-32-2</td><td>negative, positive</td></tr>
<tr><td>positive</td> <td>0.999</td> <td>AdapterHub/bert-base-uncased-pf-sst2</td><td>negative, positive</td></tr>
<tr><td>positive</td> <td>0.917</td> <td>SetFit/deberta-v3-large__sst2__train-32-1</td><td>negative, positive</td></tr>
<tr><td>positive</td> <td>0.999</td> <td>moshew/tiny-bert-aug-sst2-distilled</td><td>negative, positive</td></tr>
<tr><td>positive</td> <td>0.651</td> <td>rohansingh/autonlp-Fake-news-detection-system-29906863</td><td>negative, positive</td></tr>
<tr><td>5 stars</td> <td>0.872</td> <td>tomato/sentiment_analysis</td><td>1 star, 2 stars, 3 stars, 4 stars, 5 stars</td></tr>
<tr><td>5 stars</td> <td>0.424</td> <td>cmarkea/distilcamembert-base-sentiment</td><td>1 star, 2 stars, 3 stars, 4 stars, 5 stars</td></tr>
<tr><td>5 stars</td> <td>0.872</td> <td>nlptown/bert-base-multilingual-uncased-sentiment</td><td>1 star, 2 stars, 3 stars, 4 stars, 5 stars</td></tr>
</table>
<table>
<tr><td colspan="4" style="background-color:yellow;"> <i>I love you</i> is <b>love</b> and/or <b>joy<b></td></tr>
<tr> <th> <b>Predicted Label</b></th> <th><b>Score</b></th> <th><b>Model</b> </th> <th> <b>Labels for Model</b></th></tr>
<tr><td>joy</td> <td>0.826</td> <td>philschmid/deberta-v3-xsmall-emotion</td><td>anger, fear, joy, love, sadness, surprise</td></tr>
<tr><td>love</td> <td>0.681</td> <td>AdapterHub/roberta-base-pf-emotion</td><td>sadness, joy, love, anger, fear, surprise</td></tr>
<tr><td>joy</td> <td>0.786</td> <td>philschmid/MiniLMv2-L6-H384-emotion</td><td>sadness, joy, love, anger, fear, surprise</td></tr>
<tr><td>love</td> <td>0.649</td> <td>bhadresh-savani/roberta-base-emotion</td><td>sadness, joy, love, anger, fear, surprise</td></tr>
<tr><td>love</td> <td>0.935</td> <td>bhadresh-savani/albert-base-v2-emotion</td><td>anger, fear, joy, love, sadness, surprise</td></tr>
<tr><td>love</td> <td>0.960</td> <td>marcelcastrobr/sagemaker-distilbert-emotion-2</td><td>sadness, joy, love, anger, fear, surprise</td></tr>
</table>
<table>
<tr><td colspan="4" style="background-color:yellow;"> <i>I love you</i> is <b>fake news<b></td></tr>
<tr> <th> <b>Predicted Label</b></th> <th><b>Score</b></th> <th><b>Model</b> </th> <th> <b>Labels for Model</b></th></tr>
<tr><td>Fake</td> <td>0.998</td> <td>yaoyinnan/bert-base-chinese-covid19</td><td>Neutral, Fake, Real</td></tr>
<tr><td>Fake</td> <td>0.986</td> <td>yaoyinnan/roberta-fakeddit</td><td>Fake, Real</td></tr>
<tr><td>fake</td> <td>0.958</td> <td>Qiaozhen/fake-news-detector</td><td>real, fake</td></tr>
<tr><td>FAKE</td> <td>0.959</td> <td>Narrativaai/fake-news-detection-spanish</td><td>REAL, FAKE</td></tr>
</table>
<table>
<tr><td colspan="4" style="background-color:yellow;"> <i>I love you</i> is both <b>spam</b> and <b>ham</b></td></tr>
<tr> <th> <b>Predicted Label</b></th> <th><b>Score</b></th> <th><b>Model</b> </th> <th> <b>Labels for Model</b></th></tr>
<tr><td>spam</td> <td>0.826</td> <td>SetFit/distilbert-base-uncased__enron_spam__all-train</td><td>ham, spam</td></tr>
<tr><td>not spam</td> <td>1.000</td> <td>sureshs/distilbert-large-sms-spam</td><td>not spam, spam</td></tr>
</table>
<table>
<tr><td colspan="4" style="background-color:yellow;"> <i>I love you</i> is (mostly) not <b>hateful/offensive</b></td></tr>
<tr> <th> <b>Predicted Label</b></th> <th><b>Score</b></th> <th><b>Model</b> </th> <th> <b>Labels for Model</b></th></tr>
<tr><td>not-hate</td> <td>0.974</td> <td>aXhyra/demo_hate_1234567</td><td>not-hate, hate</td></tr>
<tr><td>neither</td> <td>0.349</td> <td>SetFit/distilbert-base-uncased__hate_speech_offensive__train-16-9</td><td>hate speech, offensive language, neither</td></tr>
<tr><td>not-hate</td> <td>0.990</td> <td>aXhyra/presentation_hate_31415</td><td>not-hate, hate</td></tr>
<tr><td>no hate speech</td> <td>0.885</td> <td>SetFit/distilbert-base-uncased__ethos_binary__all-train</td><td>no hate speech, hate speech</td></tr>
<tr><td>not-hate</td> <td>0.995</td> <td>aXhyra/hate_trained_42</td><td>not-hate, hate</td></tr>
<tr><td>hateful</td> <td>0.040</td> <td>pysentimiento/robertuito-hate-speech</td><td>hateful, targeted, aggressive</td></tr>
<tr><td>offsenive language</td> <td>0.336</td> <td>SetFit/distilbert-base-uncased__hate_speech_offensive__train-32-1</td><td>hate speech, offensive language, neither</td></tr>
<tr><td>offensive</td> <td>1.000</td> <td>simjo/model1_test</td><td>not offensive, offensive</td></tr>
<tr><td>OFFENSIVE</td> <td>0.546</td> <td>seanbenhur/tanglish-offensive-language-identification</td><td>NOT-OFFENSIVE, OFFENSIVE</td></tr>
<tr><td>neither</td> <td>0.365</td> <td>SetFit/distilbert-base-uncased__hate_speech_offensive__train-8-6</td><td>hate speech, offensive language, neither</td></tr>
<tr><td>offensive language</td> <td>1.000</td> <td>simjo/dummy-model</td><td>not offensive, offensive</td></tr>
<tr><td>hate speech</td> <td>0.350</td> <td>SetFit/distilbert-base-uncased__hate_speech_offensive__train-8-7</td><td>hate speech, offensive language, neither</td></tr>
</table>


<p>
Design goals/benefits of higher level languages such as <i>gft</i>:
<ol>

<li> Hide complexity: <i>gft</i> programs should be short (1-line) and easy to read. </li>

<li> Avoid special cases (especially in code that is exposed to users): Standard examples
such as <a href="https://github.com/huggingface/transformers/tree/master/examples">these</a>
and <a href="https://github.com/PaddlePaddle/PaddleNLP">these</a>
and longer than <i>gft</i> programs.  In many cases, 500 lines of pytorch code
can be reduced to a single line of <i>gft</i> code.  
These 500 lines of code contain many details that users do not need to know about
such as data loading, gradient descent training, and much more.
Many of these examples are very similar to one another.  Avoid duplication in
code that is exposed to large numbers of users.
</li>

<li> Code re-use: The standard examples are full of opportunities for code reuse.
The user is expected to fork the code in these examples and modify them as needed if they
want the examples to work on slightly different tasks, or slightly
different datasets.  When users modify the 500 lines of code, there will
introduce bugs.  Code reuse is safer than
editing examples.  Since the <i>gft</i> tools are based closely on these
examples, they should produce similar results, with similiar
computational resources (space and time), since both solutions are
basically running the same algorithms (and much of the same code).
</li>

<li> Flexibility/Generality: Support most datasets and models
published on hubs (HuggingFace, PaddleNLP).  The prefixes, <i>H</i>, <i>P</i> and <i>C</i>,
refer to HuggingFace, PaddleNLP and custom (local filesystem).  You
should be able to mix and match models and datasets from different
sources (<a href="https://huggingface.co/">HuggingFace</a>, <a
href="https://www.paddlepaddle.org.cn/hublist">PaddleHub/PaddleNLP</a>,
<a href="https://adapterhub.ml/">Adapter Hub</a>, etc.)  There are
currently about 30k models and 3k datasets on these hubs.  <i>gft</i> hides
complexities such as different formats for models from different
suppliers, and different types of auto classes for different purposes.
For example, users should not need to know about adapter models, and
how they are different from other types of models.</li>
</ol>





<h1 id="inference">Inference (with more details)</h1>

<i>gft_predict</i> reads from stdin and applies almost any input to almost any model.
See documentation on <a href="https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/pipelines">HuggingFace pipelines</a>
<strike>
and <a href="https://github.com/PaddlePaddle/PaddleNLP#taskflow%E5%BC%80%E7%AE%B1%E5%8D%B3%E7%94%A8%E7%9A%84%E4%BA%A7%E4%B8%9A%E7%BA%A7nlp%E8%83%BD%E5%8A%9B">PaddleNLP taskflow</a>
</strike>
for more information on </i>--task</i> argument.
<p>
Example of usage of inference scripts:

<ol>
<li><a href="#TC">text-classification </a> : The left hand side (lhs) of the equation is a single variable over classes. </li>
<li><a href="#tokenClassification">token-classification </a> : The lhs has a class variable for each token. </li>
<li><a href="#MT"> translation </a>: Machine Translation </li>
<li><a href="#fill"> fill-mask </a>: Replace "<mask>" with words. </li>
<li><a href=#QA"> question-answering </a>: Example: SQuAD.  The answer is a span (substring) of the input. 
The lhs has two class variables for each position, indicating the start and end of answer spans. </li>
<li><a href="#IC">image-classification</a>: Like text-classification, except the rhs is a picture (as opposed to text).</li>
<li><a href="#ASR">automatic-speech-recognition</a>: ASR</li>
<li><a href="#TG">text-generation</a>: Input prompt and output completion.</li>
</ol>


<h2 id="TC">Text Classification</h2>

```sh
# text classification

# example with --task argument (HuggingFace pipelines do different things with different task arguments)
echo 'I love you.' | gft_predict --model H:AdapterHub/bert-base-uncased-pf-emotion --task H:text-classification 2>/dev/null
# I love you.	love	0.6005669236183167
```

If you don't specify a model, one will be chosen for you (remove the /dev/null bits to see that distilbert-base-uncased-finetuned-sst-2-english is the default model).
Different models produce different classifications.  The default model produces positive and negative labels (sentiment).

```sh
echo 'I love you.' | gft_predict --task H:text-classification 2>/dev/null
# I love you.	POSITIVE	0.9998705387115479
echo 'I hate you.' | gft_predict --task H:text-classification 2>/dev/null
# I hate you.	NEGATIVE	0.9992952346801758
```

If you don't specify a <i>--task</i>, the class labels will be numeric, and the last field will
be a list of logits.  The class label is the argmax of the logits.

```sh
# default arguments: input is assigned to class 2 of 6 (number of classes is part of the model which was fine-tuned on data with 6 classes)
echo 'I love you.' | gft_predict --model H:AdapterHub/bert-base-uncased-pf-emotion 2>/dev/null
# I love you.	2	-0.2438915|4.8194537|5.235088|-1.7891347|-4.2359033|-5.1401916

echo 'I love you.' | gft_predict --model H:distilbert-base-uncased-finetuned-sst-2-english
# I love you.	1	-4.294976|4.6575847

echo 'I love you.' | gft_predict --model H:distilbert-base-uncased-finetuned-sst-2-english
# I hate you	0	3.8723779|-3.1543205
```
<h2 id="tokenClassification">Token Classification</h2>

```sh
echo 'I love New York.' | gft_predict --task H:token-classification --model vblagoje/bert-english-uncased-finetuned-pos 2>/dev/null
# I love New York.	i/PRON:0.9995	love/VERB:0.9989	new/PROPN:0.9986	york/PROPN:0.9988	./PUNCT:0.9997

echo 'I love New York.' | gft_predict --task H:token-classification 2>/dev/null
# I love New York.	New/I-LOC:0.9989	York/I-LOC:0.9974
```

<h2 id="MT">Machine Translation</h2>
```sh
# more examples of --task argument

# Machine Translation
# Language pair is encoded in the model; there are models in HuggingFace for many language pairs
echo 'I love you.' | gft_predict --task H:translation --model H:Helsinki-NLP/opus-mt-en-fr 2>/dev/null
# I love you.	Je t'aime.
echo 'I love you.' | gft_predict --task H:translation --model H:Helsinki-NLP/opus-mt-en-zh 2>/dev/null
I love you.	我爱你
```

<h2 id="fill"> Fill Mask </h2>
```sh
# fill mask: replace <mask> with n-best words
echo 'I <mask> you.' | gft_predict --task H:fill-mask 2>/dev/null
# I <mask> you.	 salute|0.241	 miss|0.177	 love|0.147	 thank|0.060	 applaud|0.047
```

<h2 id="QA">Question Answering</h2>

```sh
# Question Answering (SQuAD)
# Extract one example from the SQuAD dataset
gft_dataset --data H:squad --eqn 'classify_spans: answers ~ question + context'  --split val | sed 1q > /tmp/x

# Run inference on this example (and show the first 150 characters of each field on separate lines)
gft_predict --task H:question-answering --model H:$model < /tmp/x | tr '\t' '\n' | cut -c1-150
# Which NFL team represented the AFC at Super Bowl 50?|Super Bowl 50 was an American football game to determine the champion of the National Football Le
# {'text': ['Denver Broncos', 'Denver Broncos', 'Denver Broncos'], 'answer_start': [177, 177, 177]}
# answer: Denver Broncos

```

<h2 id="IC">Image Classification</h2>

<img src="https://images.all-free-download.com/images/graphicwebp/funny_cat_194619.webp" alt="funny cat" width=340>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg" alt="cat chonk" width=340>

```sh
# image classification
echo https://images.all-free-download.com/images/graphicwebp/funny_cat_194619.webp |
gft_predict --task H:image-classification 2>/dev/null
# https://images.all-free-download.com/images/graphicwebp/funny_cat_194619.webp	Egyptian cat|0.736	tiger cat|0.039	tabby, tabby cat|0.031	lynx, catamount|0.024	Persian cat|0.023

echo https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg | 
gft_predict --task H:image-classification 2>/dev/null
# https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg	lynx, catamount|0.433	cougar, puma, catamount, mountain lion, painter, panther, Felis concolor|0.035	snow leopard, ounce, Panthera uncia|0.032	Egyptian cat|0.024	tiger cat|0.023
```

<img src="doc/objects/images/PetImages/0.jpg" width=140>
<img src="doc/objects/images/PetImages/10000.jpg" width=140>
<img src="doc/objects/images/PetImages/10001.jpg" width=140>
<img src="doc/objects/images/PetImages/10002.jpg" width=140>
<img src="doc/objects/images/PetImages/10003.jpg" width=140>
<img src="doc/objects/images/PetImages/10004.jpg" width=140>
<img src="doc/objects/images/PetImages/1000.jpg" width=140>
<img src="doc/objects/images/PetImages/100.jpg" width=140>
<img src="doc/objects/images/PetImages/10.jpg" width=140>
<img src="doc/objects/images/PetImages/1.jpg" width=140>

```sh
gft_dataset --eqn 'classify: labels ~ file' --data H:nateraw/auto-cats-and-dogs --split train | head > /tmp/x
cat /tmp/x | gft_predict --task H:image-classification 2>/dev/null | awk -F/ '{print $NF}'
# 0.jpg	0	Egyptian cat|0.327	tiger cat|0.097	tabby, tabby cat|0.057	space heater|0.053	laptop, laptop computer|0.029
# 1.jpg	0	tabby, tabby cat|0.612	Egyptian cat|0.284	tiger cat|0.094	lynx, catamount|0.003	Siamese cat, Siamese|0.000
# 10.jpg	0	tabby, tabby cat|0.435	Egyptian cat|0.251	tiger cat|0.085	Persian cat|0.081	lynx, catamount|0.060
# 100.jpg	0	Egyptian cat|0.441	tabby, tabby cat|0.396	tiger cat|0.086	lynx, catamount|0.022	Persian cat|0.012
# 1000.jpg	0	Egyptian cat|0.369	tabby, tabby cat|0.129	tiger cat|0.107	Angora, Angora rabbit|0.053	Persian cat|0.042
# 10000.jpg	0	Egyptian cat|0.753	tabby, tabby cat|0.203	tiger cat|0.040	lynx, catamount|0.002	Siamese cat, Siamese|0.000
# 10001.jpg	0	Egyptian cat|0.321	tabby, tabby cat|0.061	Persian cat|0.053	tiger cat|0.033	bucket, pail|0.025
# 10002.jpg	0	Egyptian cat|0.581	tabby, tabby cat|0.222	tiger cat|0.096	lynx, catamount|0.044	Persian cat|0.019
# 10003.jpg	0	Egyptian cat|0.765	tabby, tabby cat|0.118	tiger cat|0.084	Siamese cat, Siamese|0.002	carton|0.002
# 10004.jpg	0	tabby, tabby cat|0.476	Egyptian cat|0.298	tiger cat|0.218	lynx, catamount|0.001	Siamese cat, Siamese|0.000
```

The results are much better if we replace the default model with a more appropriate model.

```sh
model=nateraw/vit-base-cats-vs-dogs
cat /tmp/x | gft_predict --model=$model --task H:image-classification | awk -F/ '{print $NF}'
# 0.jpg	0	cat|0.999	dog|0.001
# 1.jpg	0	cat|1.000	dog|0.000
# 10.jpg	0	cat|1.000	dog|0.000
# 100.jpg	0	cat|1.000	dog|0.000
# 1000.jpg	0	cat|1.000	dog|0.000
# 10000.jpg	0	cat|1.000	dog|0.000
# 10001.jpg	0	cat|1.000	dog|0.000
# 10002.jpg	0	cat|1.000	dog|0.000
# 10003.jpg	0	cat|0.999	dog|0.001
# 10004.jpg	0	cat|1.000	dog|0.000
```


<img src="objects/images/beans/healthy_test.20.jpg" width=140>
<img src="objects/images/beans/healthy_test.21.jpg" width=140>
<img src="objects/images/beans/healthy_test.22.jpg" width=140>
<img src="objects/images/beans/healthy_test.23.jpg" width=140>
<img src="objects/images/beans/healthy_test.33.jpg" width=140>
<img src="objects/images/beans/healthy_test.34.jpg" width=140>
<img src="objects/images/beans/healthy_test.35.jpg" width=140>
<img src="objects/images/beans/healthy_test.36.jpg" width=140>
<img src="objects/images/beans/healthy_test.37.jpg" width=140>
<img src="objects/images/beans/healthy_test.8.jpg" width=140>


```sh
gft_dataset --eqn 'classify: labels ~ image_file_path' --data H:beans | head  > /tmp/x
cat /tmp/x | gft_predict --task H:image-classification 2>/dev/null | awk -F/ '{print $NF}'
# healthy_test.21.jpg	2	fig|0.696	cucumber, cuke|0.013	pot, flowerpot|0.009	custard apple|0.007	leaf beetle, chrysomelid|0.005
# healthy_test.35.jpg	2	bell pepper|0.094	leaf beetle, chrysomelid|0.065	cucumber, cuke|0.058	head cabbage|0.049	ant, emmet, pismire|0.022
# healthy_test.34.jpg	2	cucumber, cuke|0.156	head cabbage|0.074	pot, flowerpot|0.022	ear, spike, capitulum|0.021	corn|0.010
# healthy_test.20.jpg	2	fig|0.848	pot, flowerpot|0.005	custard apple|0.003	jackfruit, jak, jack|0.002	cucumber, cuke|0.002
# healthy_test.36.jpg	2	custard apple|0.072	pot, flowerpot|0.057	fig|0.033	wool, woolen, woollen|0.025	necklace|0.013
# healthy_test.22.jpg	2	pick, plectrum, plectron|0.030	shower cap|0.012	leaf beetle, chrysomelid|0.011	head cabbage|0.008	spatula|0.007
# healthy_test.23.jpg	2	leaf beetle, chrysomelid|0.172	cucumber, cuke|0.049	ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle|0.043	corn|0.031	bell pepper|0.029
# healthy_test.37.jpg	2	cucumber, cuke|0.235	head cabbage|0.021	zucchini, courgette|0.015	fig|0.014	corn|0.012
# healthy_test.8.jpg	2	leaf beetle, chrysomelid|0.051	cucumber, cuke|0.045	head cabbage|0.023	ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle|0.010	fig|0.009
# healthy_test.33.jpg	2	cucumber, cuke|0.152	leaf beetle, chrysomelid|0.111	lacewing, lacewing fly|0.031	zucchini, courgette|0.030	fig|0.022
```

<h2 id="ASR">Speech Recognition</h2>

```sh
# speech recognition
gft_dataset --eqn 'ctc: text ~ file' --data H:timit_asr | head  > /tmp/x
cat /tmp/x | gft_predict --task H:automatic-speech-recognition 2>/dev/null | awk -F/ '{print $NF}'
# SX139.WAV	The bungalow was pleasantly situated near the shore.	THE BUNGALOW WAS PLEASANTLY SITUATED NEAR THE SHORE
# SA2.WAV	Don't ask me to carry an oily rag like that.	DON'T ASK ME TO CARRY AN OILY RAG LIKE THAT
# SX229.WAV	Are you looking for employment?	ARE YOU LOOKING FOR EMPLOYMENT
# SA1.WAV	She had your dark suit in greasy wash water all year.	SHE HAD YOUR DARK SUIT AND GREASY WASHWATER ALL YEAR
# SX49.WAV	At twilight on the twelfth day we'll have Chablis.	AT TWILIGHT ON THE TWELFTH DAY WE'LL HAVE CHABLI
# SX409.WAV	Eating spinach nightly increases strength miraculously.	EATING SPINACH NIGHTLY INCREASES STRENGTH MIRACULOUSLY
# SI1759.WAV	Got a heck of a buy on this, dirt cheap.	GOT A HECK OF A BY ON THIS DIRT CHEAP
# SI499.WAV	The scalloped edge is particularly appealing.	THE SCALLOPED EDGE IS PARTICULARLY APPEALING
# SX319.WAV	A big goat idly ambled through the farmyard.	A BIG GOAT IDLY AMBLED THROUGH THE FARMYARD
# SI1129.WAV	This group is secularist and their program tends to be technological.	THIS GROUP IS SECULARIST AND THEIR PROGRAMM TENDS TO BE TECHNOLOGICAL
```

<h2 id="TG">Text Generation</h2>

```sh
# text generation
echo 'A robin is a' | gft_predict --task H:text-generation --max_length 15 --num_return_sequences 1 2>/dev/null
# A robin is a	A robin is a cat or dog that has trouble keeping up with its

# NOTE: non-determinism; same prompt --> different completions
echo 'A robin is a' | gft_predict --task H:text-generation --max_length 15 --num_return_sequences 1 2>/dev/null
# A robin is a	A robin is a small bird that runs at a range where its legs
```

<p>

More examples of inference are <a href="examples/inference_examples">here</a>.  Lots of
examples on <a href="https://gluebenchmark.com/">GLUE</a> are <a
href="examples/inference_examples/model.HuggingFace/language/data.HuggingFace/their_models/glue">here</a>.

<h1>Fine-Tuning</h1>
Example of usage (of fine-tuning):

```sh
export datasets=$gft/datasets
outdir=/tmp/cola/cpkt
sh $gft/examples/fine_tuning_examples/model.HuggingFace/language/data.HuggingFace/glue/cola.sh $outdir
```

<p>
All of the shell scripts under fine_tuning_examples take a single argument (a directory for the results).

<p>
The shell scripts under model.HuggingFace use models from
HuggingFace, and shell scripts under model.PaddleHub use models from
PaddleHub and/or PaddleNLP.
Similarly, shell scripts under data.HuggingFace use datasets from
HuggingFace, and shell scripts under data.PaddleHub use datasets from
PaddleHub and/or PaddleNLP.

<p>
To run all fine-tuning examples:

```
# run all examples
cd $gft/examples/fine_tuning_examples

find . -name '*.sh' |
while read f
do
b=$checkpoints/`dirname $f`/`basename $f .sh`
sh $f $b/ckpt
done
```

<h1>Citations, Documentation, etc.</h1>

Paper (draft) is <a href="https://www.overleaf.com/read/kmhzfmjhnmfk">here.</a>
