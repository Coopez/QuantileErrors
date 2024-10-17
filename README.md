### Run

Run `main.py` for the latest combination of LSTM and lattice.

### Environment
pytorch-lattice supports at most `python=3.11`.

Required packages can be found in the `env.txt` files. Note that pytorch-lattice only supports pip, even when using conda.

The required packages are:

```plaintext
numpy
pytorch
pandas
neptune
matplotlib
seaborn
pytorch-lattice
```

### Neptune

Neptune can be used by setting `_LOG_NEPTUNE = True` in `config.py`. 
If uploading to neptune, one needs a repo and an api key to that repo. 
I import that api key over the untracked file `api_key.py`. If you want to upload tests in my neptune repo I can invite you in which case you will generate your own api key.

### Deterministic Optimization

Can be enabled in config. Not guaranteed better results, but interesting. Will set epoch to 1 automatically.
Requires packages `scipy` and `pytorch-minimize` from https://github.com/gngdb/pytorch-minimize