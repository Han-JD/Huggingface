# Utilizing early stopping in Huggingface

1. Should set metric_for_best_model on the TrainingArgument. The metric should be name of a metric returned by the evaluation with or without the prefix "eval_". [link to EarlyStoppingCallback](https://huggingface.co/docs/transformers/main_classes/callback#transformers.EarlyStoppingCallback),  [link to TrainingArguments](https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.TrainingArguments.metric_for_best_model)

    ~~~
    args = TrainingArguments(
        ...
        metric_for_best_model='cer',
        ...
        )
    ~~~
2. When the metric_for_best_model is not loss, set greater_is_better to specify if better models should have a greater metric or not. [link](https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.TrainingArguments.greater_is_better)

    ~~~
    args = TrainingArguments(
        ...
        metric_for_best_model='cer',
        greater_is_better = False,
        ...
        )
    ~~~
3. Import EarlyStoppingCallback as

    <code> from transformers import EarlyStoppingCallback </code>
4. Put EarlyStoppingCallback in callbacks when set a trainer. See more detail in the [link](https://huggingface.co/docs/transformers/main_classes/callback#transformers.EarlyStoppingCallback).

    ~~~
    trainer = Trainer(
        model,
        args,
        ...
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
    )
    ~~~

In adddition,
it is more common to set a early stopping process with the following options in the TrainingArguments. 

~~~
save_total_limit=5,
load_best_model_at_end=True,
evaluation_strategy ='steps',  
~~~

However, as a model evaluated and saved on steps, which mean the model saved before fully traind from the training dataset, the saved model would not perform as the best value showed. It would be good to test 'save_strategy="epoch"' option in the TrainingArguments.

Part for the code would looks like

~~~
from transformers import EarlyStoppingCallback
...
...
# Defining the TrainingArguments() arguments
args = TrainingArguments(
   f"training_with_callbacks",
   evaluation_strategy ='steps',
   eval_steps = 50, # Evaluation and Save happens every 50 steps
   save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
   learning_rate=2e-5,
   per_device_train_batch_size=batch_size,
   per_device_eval_batch_size=batch_size,
   num_train_epochs=5,
   weight_decay=0.01,
   push_to_hub=False,
   metric_for_best_model = 'f1',
   load_best_model_at_end=True)

trainer = Trainer(
    model,
    args,
    ...
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)
~~~

in my case

~~~
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

training_args = TrainingArguments(
    ...
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    ...
    metric_for_best_model='cer', # early stopping
    greater_is_better = False,   # lower cer is better 
    save_total_limit=2,          # save model
    load_best_model_at_end=True, 
    ...
    )

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    compute_metrics = compute_metrics,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=processor.feature_extractor,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
)

~~~

### ref
1) https://stackoverflow.com/questions/69087044/early-stopping-in-bert-trainer-instances
2) https://huggingface.co/docs/transformers/main_classes/callback#transformers.EarlyStoppingCallback
3) https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.TrainingArguments.greater_is_better 