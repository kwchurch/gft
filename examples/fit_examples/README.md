# Examples of gft_fit

For documentation on <i>gft_fit</i>, see <a href="../../doc/sections/functions/fit.md">here</a>.
<p>
Set the environment variable, <i>gft_checkpoints</i>, to some place with plenty of free disk space.
<p>
All of the *.sh files under here can be run with one argument (the output directory)

```sh
find . -name '*.sh' |
while read f
do
sh $f $gft_checkpoints/ckpt
done
```
