# gft (general fine-tuning): A Little Language for Deepnets

1-line programs for fine-tuning and inference

<h1>Installation</h1>

If you have docker

```sh
docker pull kchurch4/gft:latest
```

Alternatively,

```sh
pip install gft
```

You may want to set up the <i>gft</i> environment variable so these
work.  To do that, you may need to clone this repo since pip does not
install these extra resources.

```sh
git clone https://github.com/kwchurch/gft
ls $gft/examples
ls $gft/datasets
ls $gft/doc
```

<b style="background-color:yellow;">Unfortunately</b>, there are a number of incompatibilities between adapters, paddlespeech and
the latest version of HuggingFace transformers. There are a number of versions of requirements.txt in the requirements directory.
We recommend setting up several different virtual environments to work around some of these incompatibilities, if you want
to use some of these more advanced features.

The scripts in the examples directory will create results under
<i>$gft_checkpoints</i>.  Please set that variable to some place where
you have plenty of free disk space.  The results are large because
most fine-tuning examples copy a pre-trained model.  Given there are
many dozens of such examples, there will be many dozens of copies of
large models.

<p>
WARNING: Some of the fine-tuning scripts take a long time, and not all examples are working (yet).

<p>
<b><i>NOTE</i><b>: It is not possible to run everything under a single virtual environment.  See <a href="../../requirements/README.me">here</a> for more details.

<p>
Warning, <i>gft</i> programs will load models and datasets into your cache (typically, $HOME/.cache/huggingface and $HOME/.paddlenlp).
If you have limited disk space in your home directory, you might want to use symbolic links to point them to some place with more disk space.

<p>
If you want to use $gft/datasets/VAD, see <a href="../../datasets/VAD">instructions for reconstituing VAD</a>.