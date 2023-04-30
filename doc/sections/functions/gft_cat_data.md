# gft_cat_data

<i>gft_cat_data</i> outputs a dataset to <i>stdout</i>.  If there is an equation, then <i>gft_cat_data</i>
will output the columns mentioned in the equation.  Otherwise, all columns will be output, as illustrated
by the following two examples:

```sh
gft_cat_data --data H:dair-ai/emotion 2>/dev/null | sed 3q
```
<pre>
text|im feeling rather rotten so im not very ambitious right now	label|0
text|im updating my blog because i feel shitty	label|0
text|i never make her separate from me because i don t ever want her to feel like i m ashamed with her	label|0
</pre>

```sh
gft_cat_data --data H:dair-ai/emotion --eqn 'classify:label~text' 2>/dev/null | sed 3q
```
<pre>
im feeling rather rotten so im not very ambitious right now	0
im updating my blog because i feel shitty	0
i never make her separate from me because i don t ever want her to feel like i m ashamed with her	0
</pre>


```sh
gft_cat_data --data H:beans --split train --eqn 'classify_images:labels ~ image_file_path' | head
```

```sh
gft_cat_data --data H:timit_asr --split train --eqn 'ctc:text ~ file'
```


