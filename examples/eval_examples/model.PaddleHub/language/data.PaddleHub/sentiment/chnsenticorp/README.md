# Results with chnsenticorp

<table>
<tr><th>Model</th><th>Accuracy</th><th>f1</th><th>secnds</th></tr>
<tr><td>ernie-1.0</td><td>0.938</td><td>0.938</td><td>393</td></tr>
<tr><td>ernie-gram-zh</td><td>0.920</td><td>0.920</td><td>405</td></tr>
<tr><td>skep_ernie_1.0_large_ch</td><td>0.917</td><td>0.917</td><td>1257</td></tr>
<tr><td>ernie-tiny</td><td>0.916</td><td>0.916</td><td>172</td></tr>
<tr><td>ernie-2.0-en-finetuned-squad</td><td>0.506</td><td>0.336</td><td>380</td></tr>
<tr><td>ernie-2.0-en</td><td>0.494</td><td>0.331</td><td>406</td></tr>
</table>

<p>
Note that accuracy is about 50% for English models, and more than 90%
for Chinese models.  This is to be expected since
the text in chnsenticorp is in Chinese.  Chance accuracy for this
task is 50%.
<p>

Fine-tuning did not succeed (so far) for the following models:
ernie-2.0-large-en,
ernie-doc-base-en,
ernie-doc-base-zh,
ernie-gen-base-en,
ernie-gen-large-en-430g,
ernie-gen-large-en,
skep_ernie_2.0_large_en.

Failure modes:
<ol>
<li>Fine-tuning is still running for some large models:
ernie-2.0-large-en, skep_ernie_2.0_large_en</li>
<li>The load failed for 5 models in <b>AutoModelForSequenceClassification.from_pretrained(model_key, return_dict=True, num_labels=num_labels)</b>:
<ol>
<li><b>TypeError: __init__() missing 1 required positional argument: 'num_classes'</b>
for:
ernie-doc-base-en and
ernie-doc-base-zh.  <p><b>Recommendation</b>: We should fix this error.</li>
<li><b>AttributeError: module 'paddlenlp.transformers.ernie_gen.modeling' has no attribute 'ErnieGenForSequenceClassification'</b> for:
ernie-gen-base-en,
ernie-gen-large-en-430g and
ernie-gen-large-en.  <p><b>Recommendation</b>: we should modify the <a href="https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html">documentation</a> to make it clear which models are appropriate for fine-tuning for which tasks.</li>
</ol></li>
</ol>


