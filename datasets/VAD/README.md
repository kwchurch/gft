# splits of NRC-VAD Lexicon (https://saifmohammad.com/WebPages/nrc-vad.html)

Unfortunately, we are not allowed to redistribute the NRC-VAD lexicon.
Please download the lexicon from <a
href="https://saifmohammad.com/WebPages/nrc-vad.html">here</a>.
Set the environment variable VAD to your copy of the VAD lexicon.

```sh
cd $gft/datasets/VAD; sh reconstitute.sh
```

That will
use VAD_splits.txt to create three csv files in this directory called:
VAD.test, VAD.train, VAD.val.  The three files should look like this:

<pre>
==> VAD.test <==
Word,Valence,Arousal,Dominance
aaaaaaah,0.479,0.606,0.291
abreast,0.635,0.380,0.519
absorption,0.609,0.422,0.446
abuse,0.071,0.873,0.469
account,0.531,0.347,0.472
accumulate,0.573,0.536,0.607
aching,0.053,0.880,0.348
acidity,0.156,0.663,0.360
acknowledge,0.786,0.370,0.661

==> VAD.train <==
Word,Valence,Arousal,Dominance
aaaah,0.520,0.636,0.282
aardvark,0.427,0.490,0.437
aback,0.385,0.407,0.288
abacus,0.510,0.276,0.485
abalone,0.500,0.480,0.412
abandon,0.052,0.519,0.245
abandoned,0.046,0.481,0.130
abashed,0.177,0.644,0.307
abate,0.255,0.696,0.604

==> VAD.val <==
Word,Valence,Arousal,Dominance
abandonment,0.128,0.430,0.202
abbey,0.580,0.367,0.444
abbreviation,0.469,0.306,0.345
abeyance,0.330,0.510,0.292
abiding,0.796,0.327,0.750
abrasive,0.392,0.610,0.593
absence,0.153,0.235,0.266
absolute,0.526,0.510,0.827
absolution,0.715,0.500,0.664
</pre>

There are two more splits files, VAD.small_splits.txt and VAD.tiny_splits.txt,
that are similar to VAD_splits.txt.   You should be able to create the following files from them:
VAD.small.train, VAD.small.val, VAD.small.test, VAD.tiny.train, VAD.tiny.val, VAD.tiny.test.
 
If you have successfully reconstituted VAD.train, VAD.val and VAD.test, then

```sh
gft_summary --data C:$gft/datasets/VAD/VAD 2>/dev/null
```
should produce:
<pre>
dataset: /mnt/home/kwc/gft7/gft/datasets/VAD/VAD	No info from HuggingFace
dataset: /mnt/home/kwc/gft7/gft/datasets/VAD/VAD	splits: train: 16081 rows, val: 1959 rows, test: 1967 rows
dataset: /mnt/home/kwc/gft7/gft/datasets/VAD/VAD	split: train	columns: Word, Valence, Arousal, Dominance
# dataset: /mnt/home/kwc/gft7/gft/datasets/VAD/VAD --> 0 models
</pre>
This says that there are 16k rows (words) in train, 2k rows in val and 2k in test.

<p>
You can compare your results with this (which does not depend on reconstituting files):

```sh
gft_summary --data C:$gft/datasets/VAD/simple/VAD.simple.10k 2>/dev/null
```
