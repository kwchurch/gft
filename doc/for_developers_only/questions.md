# Questions for colleagues at Baidu (not for general public)

<ul>
<li>Question 1: Status: <i>unresolved</i></ul>
</ul>

I could use help with the summarize_model function in 
<a href="http://gitlab.baidu.com/kennethchurch/gft/blob/master/gft_internals/gft_summary_pd.py">gft_summary_pd</a>.

I would like to print the names of the labels, as opposed to the
number of labels, but I don't know how to get the names of the labels
from a PaddleNLP model.

```sh
gft_summary --model H:bert-base-cased 2>/dev/null
> model: bert-base-cased	model: bert-base-cased	downloads: 2971111	likes: 13	task: fill-mask
> model: bert-base-cased	labels: LABEL_0, LABEL_1

gft_summary --model P:bert-base-cased 2>/dev/null
> model: bert-base-cased	num_classes: 2
```

