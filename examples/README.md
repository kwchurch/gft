# gft (general fine-tuning): Examples

There are hundreds of examples under this directory.
<p>
Each example is a shell file (ending with the suffix ".sh").
<p>
The fit examples all take an argument for the output_directory.
The other examples take no arguments.
<p>
There are 4 subdirectories for the 4 functions.  There is a README in each of these 4 subdirectories:

<ol>
<li><a href="fit_examples">fit_examples</a></li>
<li><a href="predict_examples">predict_examples</a></li>
<li><a href="summary_examples">summary_examples</a></li>
<li><a href="eval_examples">eval_examples</a></li>
</ol>

Naming conventions:

<ul>
<li>model.HuggingFace: models are from HuggingFace</li>
<li>model.PaddleHub: models are from PaddeNLP/PaddleHub</li>
<li>our_models: models were created by gft_fit</li>
<li>their_models: models were fine-tuned by some other means and posted on a hub (often by a 3rd party)</li>
<li>data.HuggingFace: dataset is from HuggingFace (or a custom dataset)</li>
<li>data.PaddleHub: dataset is from PaddleNLP/PaddleHub</li>
<li>language: The example involves natural language (as opposed to speech)</li>
<li>speech: The example involves speech (as opposed to natural language)</li>
</ul>

It is assumed that gft_fit is in your path.
<p>
Environment Variables:

<ul>
<li>$gft: points to repo ($gft/gft_fit should be in your path)</li>
<li>$gft_checkpoints: points to a place with plenty of disk space</li>
<li>$gft/datasets: points to some custom datasets (included in repo)</li>
</ul>
