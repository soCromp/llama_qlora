Based on the QLoRA repo

Setup: `conda env create -f environment.yml`

Make sure to install huggingface version of transformers:
```shell
pip uninstall -y transformers
wget https://github.com/huggingface/transformers/archive/refs/tags/v4.39.3.tar.gz
tar -xvf v4.39.3.tar.gz
cd transformers-4.39.3
pip install -e .
```
https://huggingface.co/DiscoResearch/mixtral-7b-8expert/discussions/9

(Updating environment.yml): `conda list --explicit > environment.yml`

Bug modification: in `transformers/models/llama/modeling_llama.py`, to avoid datatype issues, change line 1075 (of `LlamaModel._update_causal_mask`) from
```python
causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
```
into
```python
if dtype == torch.bfloat16:
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=torch.float, device=device)
        else:
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
```

