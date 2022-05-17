# Questions for colleagues at Baidu (not for general public)

<ul>
<li>Question 0: Posted May 17, 2022; Status: <i>unresolved</i></li>
<li>Question 1: Status: <i>resolved</i></li>
</ul>

<h1>Question 0: Summary</h1>

Please compare fit_for_classify_hf.py with fit_for_classify_pd.py

```sh
sh $gft/examples/fit_examples/model.HuggingFace/language/data.HuggingFace/glue/mnli_matched.sh $gft_checkpoints/junk/mnli_matched_hf
sh $gft/examples/fit_examples/model.PaddleHub/language/data.HuggingFace/glue/mnli_matched.sh $gft_checkpoints/junk/mnli_matched_hf
```

The hf version works, but the pd version fails.  Here is the output from the pd version just before the break point (search for pdb.set_trace in fit_for_classify_pd.py):

```sh
label_list: [0, 1, 2]
epoch: 0
num_labels: 3
logits: Tensor(shape=[8, 2], dtype=float32, place=Place(gpu:0), stop_gradient=False,
       [[-0.68304992, -0.53298235],
        [-0.18006738, -1.16934550],
        [-0.11677605, -0.95094025],
        [ 0.08184259, -1.03752649],
        [-0.30261040, -0.68470716],
        [ 0.02099895, -1.03713500],
        [-0.17123263, -0.91750258],
        [-0.34823859, -0.86686128]])
labels: Tensor(shape=[8, 1], dtype=int64, place=Place(gpu:0), stop_gradient=True,
       [[2],
        [0],
        [1],
        [2],
        [0],
        [1],
        [2],
        [0]])
successes: 0
> /mnt/home/kwc/public_github/gft/gft/gft_internals/fit_for_classify_pd.py(340)fit()
-> loss.backward()
(Pdb)
```

Here is the output from the hf version (search for "uncomment these lines to see details" and then uncomment the lines after that).

```sh
outputs: SequenceClassifierOutput(loss=tensor(1.0791, device='cuda:0', grad_fn=<NllLossBackward0>), logits=tensor([[ 0.1724, -0.1555,  0.0753],
        [ 0.2503, -0.0548, -0.2671],
        [-0.1170, -0.1783,  0.0414],
        [-0.1821, -0.2637,  0.0041],
        [ 0.5538, -0.1977, -0.2777],
        [ 0.7887, -0.1788, -0.6303],
        [ 0.4694,  0.0098, -0.3489],
        [ 0.6773, -0.3299, -0.2744]], device='cuda:0',
       grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
loss: tensor(1.0791, device='cuda:0', grad_fn=<NllLossBackward0>)
> /mnt/home/kwc/public_github/gft/gft/gft_internals/fit_for_classify_hf.py(264)fit()
-> loss = loss / gas
(Pdb)
```

Note the difference in shapes.  The hf version creates 3 logits (because num_labels is 3), but the pd version creates only 2 logits.  Do you know why the pd version creates only 2 logits?

<h1>Question 1: Summary</h1>
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

