## Retrieval-Extraction (REX) language model pre-training for open domain QA

We provide code examples for building retrieval-extraction (REX) based language model pre-training and evaluating
the pre-trained models on Squad 2 in this repo.

## Pretraining
Start pretraining of REX LM by simpling runing
```
./experiment/rex.sh $EXPERIMENT_NAME
```

## Fine-tuning and evaluation
Finetune the pretrained model on Squad 2 by running
```
/experiment/finetune.sh $FINETUNE_EXPERIMENT_NAME
```
Note that you should specify the checkpoint you would like to use in the finetune.sh file.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for more information.
