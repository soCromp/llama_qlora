Based on the QLoRA repo

Setup: `conda env create -f environment.yml`

(Updating environment.yml): `conda list --explicit > environment.yml`

Bug modification: make line 89, the last line of LlamaRMSNorm's forward() to `return self.weight * hidden_states**.to(input_dtype).to(self.weight.device)**` 
in transformers/models/llama/modeling_llama.py to avoid datatype issues

