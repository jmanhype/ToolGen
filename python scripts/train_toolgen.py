def load_base_model():
    print("Loading base model...")
    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    # Load model without DeepSpeed first
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.float16 if not config.quantized else None,  # Only use FP16 if not quantized
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load DeepSpeed configuration
    ds_config_path = os.path.join(os.path.dirname(__file__), "ds_config.json")
    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)
    
    # Initialize DeepSpeed-enabled model
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config
    )

    return model_engine, tokenizer