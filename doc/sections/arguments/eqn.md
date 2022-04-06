# gft Equations

Many of the examples make use of equations such as:

```sh
--eqn 'classify: label ~ question1 + question2'
```

The keyword, classify, is a task (also known as text-classification).  See <a href="task.md">here</a>
for more discussion of tasks such as: regress, classify_tokens, classify_spans, ctc, etc.  
<p>
In the
classification case, for each input (pairs of two questions), there is
a single label (semantically similar or not).  Classify also
generalizes from binary classification to multiclass classification
(for tasks such as emotion classification).  As shown in the table below,
equations start with a number of different keywords (tasks):
<ol>
<li> <i>classify</i>: lhs denotes a set of classes </li>
<li> <i>regress</i> : lhs denotes a point in a vector space </li>
<li> <i>classify_tokens</i> : there is a classification task for each token on the rhs</li>
<li> <i>classify_spans</i> : used for SQuAD-like tasks where the output should be a span (substring) of the rhs </li>
<li> <i>ctc</i>: used in speech recognition where the input is audio and the output is text </li>
</ol>

There are a number of examples of equations in the table below:

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

