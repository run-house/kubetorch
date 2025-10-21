ENV KT_CLS_OR_FN_NAME torch_ddp
ENV KT_DISTRIBUTED_CONFIG {"distribution_type": "pytorch", "num_proc": 4}
ENV KT_FILE_PATH tests/assets/torch_ddp/torch_ddp.py
ENV KT_INIT_ARGS null
ENV KT_MODULE_NAME tests.assets.torch_ddp.torch_ddp
