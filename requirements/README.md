# requirements for GFT (General Fine-Tuning)

<i>gft</i> takes dependencies on many python packages, that are
constantly being updated.



<p>
Unfortunately, it is not possible to support all of these packages in a single virtual environment.  We have provided
a number of different requirements.txt files.  Please replace requirements.txt as desired

```sh
pip install $gft/requirements/requirements.txt
```

<ol>
<li><a href="requirements.txt">requirements.txt</a><li>
<li><a href="requirements.txt">requirements_without_adapters.txt</a><li>
<li><a href="requirements.txt">requirements_with_adapters.txt</a><li>
<li><a href="requirements.txt">requirements_for_paddlespeech.txt</a><li>
<ol>

<h2>Known Issues</h2>

<ol><li>
<a href="https://adapterhub.ml/">Adapters</a>
require <i>adapter-transformers</i>.  That package
requires an old version of <i>transformers</i>.  Many other features
require never versions of <i>transformers</i></li>

<li><i>paddlespeech</i> requires old versions of <i>numpy</i>, <i>pandas</i>, <i>librosa</i>
and <i>tokenizers</i></li>

<li> A number of models/datasets/tasks depend on packages such as: <i>py7zr</i>,
<i>conllu</i>,
<i>apache_beam</i> and
<i>timm</i>.</li>

<li>I have not managed to install TapasModel (required for table-question-answering).
Instructions are installing TapasModel are
<a href="https://github.com/rusty1s/pytorch_scatter.">here</a>.</li>
</ol>