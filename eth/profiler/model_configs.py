# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

## "algo" stands for tensor parallel partition algorithm
model_prof_configs = {
    "gpt": {
        "dtype": "fp16",
        "model_size": ["350M", "1_3B", "2_6B", "6_7B", "13B", "scale-layer"],
        #"mbs": [1,2,3,4,5,6,7,8],
        "mbs": [1,2,4,8],
        "algo": [0, 1]
    }
}

# model_size: (num_layers, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype)
gpt_configs = {
    "350M": (1, 2048, 1024, 1024*4, 16, 1024//16, 51200, "fp16"),
    "1_3B": (1, 2048, 2048, 2048*4, 32, 2048//32, 51200, "fp16"),
    "2_6B": (1, 2048, 2560, 2560*4, 32, 2560//32, 51200, "fp16"),
    "6_7B": (1, 2048, 4096, 4096*4, 32, 4096//32, 51200, "fp16"),
    "13B": (1, 2048, 5120, 5120*4, 40, 5120//40, 51200, "fp16"),
    "scale-layer": (1, 2048, 512, 512*4, 8, 512//8, 51200, "fp16")
}
