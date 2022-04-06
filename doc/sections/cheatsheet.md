# GFT (General Fine-Tuning) Cheatsheet

<ul>
<li>
4+1 Functions:
<ol>
<li><a href="functions/gft_fit.md">gft_fit</a>: f_pre -> f_post (fine-tuning)</li>
<li><a href="functions/gft_predict.md">gft_predict</a>: f(x) -> yhat (inference)</li>
<li><a href="functions/gft_eval.md">gft_eval</a>: Apply a model to a dataset and return a score</li>
<li><a href="functions/gft_summary.md">gft_summary</a>: Summarize (almost) anything; find ``good stuff''</li>
<li><a href="functions/gft_cat_data.md">gft_cat_data</a>: Output a dataset on <i>stdout</i></li>
</ol>
</li>

<li>
4 Arguments (to most of the functions above):
<ol>
<li>--<a href="arguments/data.md">data</a></li>
<li>--<a href="arguments/model.md">model</a></li>
<li>--<a href="arguments/eqn.md">eqn</a> (task: y ~ x1 + x2)</li>
<li>--<a href="arguments/task.md">task</a>
<ol>
<li>classify or text-classification</li>
<li>classify_tokens or token-classification</li>
<li>classify_spans, QA or question-answering</li>
<li>classify_audio or audio-classification</li>
<li>classify_images or image-classification</li>
<li>regress</li>
<li>text-generation</li>
<li>MT or translation</li>
<li>ASR, ctc or automatic-speech-recognition</li>
<li>fill-mask</li>
</ol>
</li>
</ul>


