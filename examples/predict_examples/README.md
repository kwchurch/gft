# Examples of gft_predict

For documentation on <i>gft_predict</i>, see <a href="../../doc/sections/functions/predict.md">here</a>.
<p>
All of the *.sh files under here can be run without arguments:

```sh
find . -name '*.sh' |
while read f
do
sh $f
done
```
<p>
All of these scripts output to <i>stdout</i> and <i>stderr</i>.  You may want to redirect them appropriately.

<p>
NOTE: the output can be large, since there will be a line on
stdout for each prediction (for each example in the appropriate
dataset split).

<p>
Warning, these programs will load models and datasets into your
cache (typically, $HOME/.cache/huggingface and $HOME/.paddlenlp).  If
you have limited disk space in your home directory, you might want to
use symbolic links to point them to some place with more disk space.
