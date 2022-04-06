# Examples of gft_eval

For documentation on gft_eval, see <a href="../../doc/sections/functions/eval.md">here</a>.
<p>
Some of the shell scripts assume that $gft_checkpoints has already been set,
and that models have been left there by gft_fit (see <a href="../fit_examples">here</a>).
<p>
All of the *.sh files under here can be run without arguments

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
Warning, these programs will load models and datasets into your
cache (typically, $HOME/.cache/huggingface and $HOME/.paddlenlp).  If
you have limited disk space in your home directory, you might want to
use symbolic links to point them to some place with more disk space.
