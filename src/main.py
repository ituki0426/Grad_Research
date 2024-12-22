import torch
import torchvision
import librosa
import matplotlib
import numpy
from datasets import load_dataset
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoFeatureExtractor,AutoModelForAudioClassification
import evaluate
import accelerate
import wandb
from transformers import TrainingArguments, Trainer

max_duration=30.0
feature_extractor=AutoFeatureExtractor.from_pretrained('ntu-spml/distilhubert', do_normalize=True, return_attention_mask=True)
sampling_rate=feature_extractor.sampling_rate
metric=evaluate.load('accuracy')

def preprocess_function(examples):
    audio_arrays=[x['array'] for x in examples['audio']]
    inputs=feature_extractor(audio_arrays, sampling_rate=sampling_rate, max_length=int(sampling_rate*max_duration), truncation=True, return_attention_mask=True)
    return inputs

def compute_metrics(eval_pred):
    predictions=np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def main():
    load_dotenv()
    login(token=os.environ['HF_TOKEN'])
    wandb.login(key=os.environ['WANDB_API_KEY']) # Pass your W&B API key here
    wandb.init(project=os.environ['WANDB_PROJECT']) # Add your W&B project name 

    dataset=load_dataset('marsyas/gtzan')

    dataset=dataset['train'].train_test_split(seed=42, shuffle=True, test_size=.2)
    id2label_function=dataset['train'].features['genre'].int2str

    dataset_encoded=dataset.map(preprocess_function, remove_columns=['audio', 'file'], batched=True, batch_size=100, num_proc=1)
    dataset_encoded=dataset_encoded.rename_column('genre', 'label')

    id2label={str(i): id2label_function(i) for i in range(len(dataset_encoded['train'].features['label'].names))}
    label2id={v:k for k, v in id2label.items()}
    num_labels=len(id2label)
    model=AutoModelForAudioClassification.from_pretrained('ntu-spml/distilhubert',num_labels=num_labels, label2id=label2id, id2label=id2label)

    training_args=TrainingArguments(
        output_dir=os.environ(['WANDB_NAME']),
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        learning_rate=5e-5,
        seed=42,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=100, # We control the total training steps to fit the limied resources
        num_train_epochs=2,
        warmup_ratio=0.1,
        fp16=True,
        save_total_limit=2,
        report_to='wandb',
        run_name=os.environ(['WANDB_NAME'])
    )

    trainer=Trainer(model=model, args=training_args, train_dataset=dataset_encoded['train'], eval_dataset=dataset_encoded['test'], tokenizer=feature_extractor, compute_metrics=compute_metrics)
    trainer.train()
if __name__ == '__main__':
    main()
