# Arguments for GFT (General Fine-Tuning)

The following will find examples that use is_split_into_words:

```sh
egrep -c is_split_into_words `find $gft/examples -name '*.sh'` | awk -F: '$NF > 0 {print $1}'
```

You can find examples of other arguments by replacing is_split_into_words with another argument of interest.
<p>

The following arguments are sorted by frequency of use in the examples under <a href="../../../examples">$gft/examples</a>:

<ol>
<li href="model"> --model : <a href="model.md">Doc</a></li>
<li href="data"> --data <a href="data.md">Doc</a></li>
<li href="eqn"> --eqn <a href="eqn.md">Doc</a></li>
<li href="split"> --split : import for gft_eval, gft_cat_doc, gft_predict; select one of the three splits of a dataset </li>
<li href="num_train_epochs"> --num_train_epochs : used in stopping rules; an important parameter for hyperparameter tuning </li>
<li href="output_dir"> --output_dir : used in gft_fit; location to leave the rules (post-trained model)</li>
<li href="metric"> --metric : used in gft_eval; defaults to accuracy for most tasks, but mean squared error for regression;
most HuggingFace metrics are supported:
accuracy,
adjusted_mutual_info,
adjusted_rand,
average_precision,
balanced_accuracy,
brier_score_loss,
completeness,
explained_variance,
fowlkes_mallows,
homogeneity,
jaccard,
log_loss,
max_error,
mean_absolute_error,
mean_absolute_percentage_error,
mean_gamma_deviance,
mean_poisson_deviance,
mean_squared_error,
mean_squared_log_error,
median_absolute_error,
glue,
mutual_info,
normalized_mutual_info,
r2,
rand,
recall,
roc_auc,
top_k_accuracy,
v_measure.
 </li>
<li href="splits"> --splits : three comma separated values; defaults to train,val,test (but many datasets use different names for these splits).
Some datasets do not provide a validation split.  We often use __select_from_train__ for the validation split.  In this case, a portion of the training set is used
for validation.</li>
<li href="base_model"> --base_model : important for gft_eval; this argument specifies the base model (when it is difficult to determine that from the --model argument)</li>
<li href="do_not_catch_errors"> --do_not_catch_errors : by default, gft_eval and gft_summary will catch errors; this flag will turn that off (useful for debugging)</li>
<li href="per_device_train_batch_size"> --per_device_train_batch_size : an important parameter for hyperparameter tuning</li>
<li href="figure_of_merit"> --figure_of_merit : defaults to accuracy (for most tasks), and mean squared error (for regression)</li>
<li href="per_device_eval_batch_size"> --per_device_eval_batch_size </li>
<li href="task"> --task <a href="task.md">Doc</a></li>
<li href="is_split_into_words"> --is_split_into_words : important for the classify_spans task</li>
<li href="data_dir"> --data_dir : Some datasets such as <a href="https://huggingface.co/datasets/senti_lex">senti_lex</a> need to be registered and downloaded manually.  This argument makes it possible to point to the local version of that dataset.</li>
<li href="text_field"> --text_field </li>
<li href="learning_rate"> --learning_rate : an important parameter for hyperparameter tuning</li>
<li href="warmup_steps"> --warmup_steps: an important parameter for hyperparameter tuning </li>
<li href="save_total_limit"> --save_total_limit </li>
<li href="layerdrop"> --layerdrop </li>
<li href="group_by_length"> --group_by_length </li>
<li href="freeze_feature_extractor"> --freeze_feature_extractor </li>
<li href="fp16"> --fp16 : use small floats (saves space)</li>
<li href="evaluation_strategy"> --evaluation_strategy </li>
<li href="chars_to_ignore"> --chars_to_ignore : used in ASR (speech) tasks </li>
<li href="weight_decay"> --weight_decay: an important parameter for hyperparameter tuning </li>
<li href="max_seq_length"> --max_seq_length </li>
<li href="gradient_accumulation_steps"> --gradient_accumulation_steps </li>
<li href="doc_stride"> --doc_stride </li>
<li href="cpus-per-task"> --cpus-per-task : an important parameter for hyperparameter tuning</li>
<li href="nproc_per_node"> --nproc_per_node : an important parameter for hyperparameter tuning</li>
<li href="gradient_checkpointing"> --gradient_checkpointing </li>
</ol>
