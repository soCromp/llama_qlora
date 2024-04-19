Based on the QLoRA repo

# Setup

## Environment

`conda env create -f environment.yml`

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

## Bug modification

**First**: in `transformers/models/llama/modeling_llama.py`, to avoid datatype issues, change line 1075 (of `LlamaModel._update_causal_mask`) from
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

**Second**: 
Change `transformers/modeling_utils.py` line 1164 to `nb_params = torch.tensor(1, dtype=self.hf_quantizer.quantization_config.bnb_4bit_quant_storage).element_size()`.

## Model registration

Run through `multihead.ipynb`, updating file paths where necessary.
