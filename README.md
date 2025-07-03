# Sentiment Analysis LLM

This project provides a minimal example of fine-tuning a pre-trained language model for sentiment analysis using the [Hugging Face Transformers](https://huggingface.co/docs/transformers) library.

## Requirements

Install dependencies using the helper script. Pass your proxy URL if you are
behind a corporate firewall:

```bash
# without a proxy
./setup_env.sh

# or with a proxy
./setup_env.sh http://my.proxy:3128
```

## Training

Run `train_sentiment_model.py` to fine-tune a small model on the IMDb dataset:

```bash
python train_sentiment_model.py --output_dir ./model
```

This downloads the IMDb dataset and a pre-trained DistilBERT model, then fine-tunes the model for sentiment classification. The fine-tuned model is saved to the specified `--output_dir`.

## Inference

After training, use `predict_sentiment.py` to classify new text:

```bash
python predict_sentiment.py --model_dir ./model --text "I love this movie!"
```

The script prints the predicted sentiment label (`positive` or `negative`).

## Proxy troubleshooting

Both training and inference download models and datasets from the internet.
If you are behind a network proxy, set the standard `HTTP_PROXY` and
`HTTPS_PROXY` environment variables so that `transformers` and `datasets`
can access Hugging Face servers. The `setup_env.sh` script accepts the proxy
URL as an argument to ease dependency installation.
