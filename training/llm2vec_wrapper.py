from llm2vec import LLM2Vec
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoConfig,
    PretrainedConfig,
    AutoTokenizer,

)
import torch
import logging
import json
import os
logger = logging.getLogger(__name__)
class LLM2VecWrapper(LLM2Vec):
    def __init__(self, *args, **kwargs):
        super(LLM2VecWrapper, self).__init__(*args, **kwargs)
    
    def add_lora_adapters(
        self,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=None,
        force_add=False,
        adapter_name: str = "llm2clip",
    ):
        if target_modules is None:
            # Keep it consistent across train/eval (Llama-style attention only)
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,   # ← was "CAUSAL_LM"
            inference_mode=False,                    # trainable adapters
        )

        # Apply/add LoRA by adapter name and make it active
        if isinstance(self.model, PeftModel):
            if force_add or adapter_name not in getattr(self.model, "peft_config", {}):
                self.model.add_adapter(adapter_name, lora_config)
            self.model.set_adapter(adapter_name)
        else:
            # Wrap base with PEFT; default adapter is 'default'. Optionally add and switch to named adapter.
            self.model = get_peft_model(self.model, lora_config)  # creates 'default'
            if adapter_name != "default":
                self.model.add_adapter(adapter_name, lora_config)
                self.model.set_adapter(adapter_name)
            else:
                self.model.set_adapter("default")

        # Remember active adapter name for downstream logic
        try:
            self.active_adapter = self.model.active_adapter
        except Exception:
            self.active_adapter = adapter_name

        self.model.print_trainable_parameters()
        return self.model
    
    def to(self, *args, **kwargs):
        """Match torch.nn.Module.to signature; also move latent_attn."""
        result = super().to(*args, **kwargs)

        # Infer target device / dtype from args/kwargs for latent_attn move
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        if device is None and len(args) > 0:
            device = args[0]  # could be a torch.device, str or dtype
        # If nothing was provided, fall back to the module's current device
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = None

        if hasattr(result, "latent_attn") and result.latent_attn is not None:
            if device is not None and dtype is not None:
                result.latent_attn = result.latent_attn.to(device=device, dtype=dtype)
            elif device is not None:
                result.latent_attn = result.latent_attn.to(device)
            elif dtype is not None:
                result.latent_attn = result.latent_attn.to(dtype=dtype)

        return result

    def prepare_for_tokenization(self, text):
        text = (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            + text.strip()
            + "<|eot_id|>"
        )
        return text
    
    def encode_text(self, text, max_length=None):
        """
        Encode text to embeddings with proper embed_mask handling.
        
        Args:
            text (str or list): Text(s) to encode
            max_length (int, optional): Maximum sequence length
        
        Returns:
            torch.Tensor: Text embeddings
        """
        if max_length is None:
            max_length = getattr(self, 'max_length', 512)
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        )
        
        # Add embed_mask (same as attention_mask for simple text encoding)
        inputs["embed_mask"] = inputs["attention_mask"].clone()
        
        # Move to same device as model 
        import torch
        model_device = next(self.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embeddings = self(inputs)
        
        return embeddings
    
    def tokenize_with_separator(self, texts, max_length=None, separator='!@#$%^&*()'):
        """
        Tokenize texts with special handling for separator-based splitting.
        This is useful for instruction-following tasks.
        
        Args:
            texts (list): List of texts to tokenize
            max_length (int, optional): Maximum sequence length  
            separator (str): Separator to split instruction from text
        
        Returns:
            dict: Tokenized inputs with attention masks and embed masks
        """
        if max_length is None:
            max_length = getattr(self, 'max_length', 512)
            
        texts_2 = []
        original_texts = []
        
        for text in texts:
            parts = text.split(separator)
            texts_2.append(parts[1] if len(parts) > 1 else "")
            original_texts.append("".join(parts))

        # Tokenize original texts
        tokenized = self.tokenizer(
            original_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        
        # Create embedding masks for the separated parts
        import torch
        embed_mask = None
        for t_i, t in enumerate(texts_2):
            ids = self.tokenizer(
                [t],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            
            e_m = torch.zeros_like(tokenized["attention_mask"][t_i])
            if len(ids["input_ids"][0]) > 0:
                e_m[-len(ids["input_ids"][0]):] = torch.ones(len(ids["input_ids"][0]))
                
            if embed_mask is None:
                embed_mask = e_m.unsqueeze(0)
            else:
                embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)

        tokenized["embed_mask"] = embed_mask
        return tokenized
    
    def encode_with_instruction(self, texts, max_length=None, separator='!@#$%^&*()'):
        """
        Encode texts with instruction-following using separator-based processing.
        
        Args:
            texts (list): List of texts with instructions separated by separator
            max_length (int, optional): Maximum sequence length
            separator (str): Separator between instruction and text
        
        Returns:
            torch.Tensor: Text embeddings
        """
        tokenized = self.tokenize_with_separator(texts, max_length, separator)
        
        # Move to same device as model
        import torch
        model_device = next(self.parameters()).device
        tokenized = {k: v.to(model_device) for k, v in tokenized.items()}
        
        with torch.no_grad():
            embeddings = self(tokenized)
        
        return embeddings

    def encode_with_separator(self, texts, device=None, max_length=None, separator='!@#$%^&*()'):
        """
        Encode texts with special separator-based handling for instruction/text pairs.
        
        Args:
            texts (list): List of texts to encode (with separator for instruction/text pairs)
            device: Device to run on (auto-detect if None)
            max_length: Maximum sequence length (use model default if None)
            separator: Separator string for instruction/text pairs
        
        Returns:
            torch.Tensor: Embeddings for the texts
        """
        if device is None:
            device = next(self.parameters()).device
        if max_length is None:
            max_length = 512
            
        # Ensure model is on the right device
        self = self.to(device)
        
        # Process texts with separator
        texts_2 = []
        original_texts = []
        
        for text in texts:
            parts = text.split(separator)
            texts_2.append(parts[1] if len(parts) > 1 else "")
            original_texts.append("".join(parts))

        # Tokenize original texts
        tokenized = self.tokenizer(
            original_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        
        # Create embedding masks
        embed_mask = None
        for t_i, t in enumerate(texts_2):
            ids = self.tokenizer(
                [t],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            
            e_m = torch.zeros_like(tokenized["attention_mask"][t_i])
            if len(ids["input_ids"][0]) > 0:
                e_m[-len(ids["input_ids"][0]):] = torch.ones(len(ids["input_ids"][0]))
                
            if embed_mask is None:
                embed_mask = e_m.unsqueeze(0)
            else:
                embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)

        tokenized["embed_mask"] = embed_mask
        
        # Move to device and compute embeddings
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        tokenized = {k: v.to(self.model.dtype) if v.dtype.is_floating_point else v 
                    for k, v in tokenized.items()}
        
        with torch.no_grad():
            embeddings = self(tokenized)
            
        return embeddings

    def compute_similarities(self, query_text, candidate_texts, device=None, separator='!@#$%^&*()'):
        """
        Compute similarity scores between a query text and candidate texts.
        
        Args:
            query_text (str): The query text (with separator for instruction/text pairs)
            candidate_texts (list): List of candidate texts to compare against
            device: Device to run on (auto-detect if None)
            separator: Separator string for instruction/text pairs
        
        Returns:
            torch.Tensor: Similarity scores for each candidate
        """
        import torch.nn.functional as F
        
        if device is None:
            device = next(self.parameters()).device
            
        # Combine query and candidates
        all_texts = [query_text] + candidate_texts
        
        # Get embeddings
        embeddings = self.encode_with_separator(all_texts, device=device, separator=separator)
        
        # Compute similarities between query (first embedding) and candidates
        similarities = F.cosine_similarity(embeddings[0], embeddings[1:], dim=1)
        
        return similarities

    def _load_latent_attention_weights(self, model_path, use_safetensors=True):
        """
        Automatically load latent attention weights from model files.
        
        Args:
            model_path: Path to model (local directory or HuggingFace repo)
            use_safetensors: Whether to use safetensors format
        """
        import os
        
        if os.path.isdir(model_path):
            # Local directory - try pytorch_model.bin first
            pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                print(f"Loading latent attention weights from {pytorch_model_path}")
                try:
                    import torch
                    state_dict = torch.load(pytorch_model_path, weights_only=True)
                    latent_attn_weights = {k: v for k, v in state_dict.items() if k.startswith('latent_attn.')}
                    
                    if latent_attn_weights:
                        missing_keys, unexpected_keys = self.latent_attn.load_state_dict(
                            {k.replace('latent_attn.', ''): v for k, v in latent_attn_weights.items()},
                            strict=False
                        )
                        if not missing_keys and not unexpected_keys:
                            print(f"✅ Successfully loaded {len(latent_attn_weights)} latent attention weights")
                        else:
                            print(f"⚠️ Partial loading: missing={missing_keys}, unexpected={unexpected_keys}")
                    else:
                        print("⚠️ No latent attention weights found in the model file")
                except Exception as e:
                    print(f"❌ Error loading latent attention weights: {e}")
        else:
            # HuggingFace repository - load from safetensors
            if use_safetensors:
                print("Loading latent attention weights from HuggingFace safetensors...")
                try:
                    from safetensors.torch import load_file
                    from huggingface_hub import hf_hub_download
                    
                    # Download the safetensors file
                    safetensors_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")
                    
                    # Load weights from safetensors
                    safetensors_weights = load_file(safetensors_path)
                    
                    # Extract latent attention weights
                    latent_attn_weights = {k: v for k, v in safetensors_weights.items() if k.startswith('latent_attn.')}
                    
                    if latent_attn_weights:
                        print(f"Found {len(latent_attn_weights)} latent attention weights in safetensors")
                        
                        # Load the weights into the latent attention module
                        missing_keys, unexpected_keys = self.latent_attn.load_state_dict(
                            {k.replace('latent_attn.', ''): v for k, v in latent_attn_weights.items()},
                            strict=False
                        )
                        
                        if not missing_keys and not unexpected_keys:
                            print(f"✅ Successfully loaded {len(latent_attn_weights)} latent attention weights from safetensors")
                        else:
                            print(f"⚠️ Partial loading: missing={missing_keys}, unexpected={unexpected_keys}")
                    else:
                        print("⚠️ No latent attention weights found in safetensors file")
                        
                except Exception as e:
                    print(f"❌ Error loading latent attention weights from safetensors: {e}")

    @classmethod
    def from_pretrained(
        cls,
        base_model_name_or_path,
        peft_model_name_or_path=None,
        merge_peft=False,
        enable_bidirectional=True,
        extra_model_name_or_path=None,
        **kwargs,
    ):
        # pop out encoder args
        keys = ["pooling_mode", "max_length", "doc_max_length", "skip_instruction"]
        encoder_args = {
            key: kwargs.pop(key, None) for key in keys if kwargs.get(key) is not None
        }

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(base_model_name_or_path)
        config_class_name = config.__class__.__name__

        model_class = cls._get_model_class(
            config_class_name, enable_bidirectional=enable_bidirectional
        )
        model = model_class.from_pretrained(base_model_name_or_path, **kwargs)

        if os.path.isdir(base_model_name_or_path) and os.path.exists(
            f"{base_model_name_or_path}/config.json"
        ):
            with open(f"{base_model_name_or_path}/config.json", "r") as fIn:
                config_dict = json.load(fIn)
            config = PretrainedConfig.from_dict(config_dict)
            model.config._name_or_path = config._name_or_path

        # For special case where config.json and adapter weights are in the same directory
        if hasattr(model, "peft_config"):
            model = PeftModel.from_pretrained(
                model,
                base_model_name_or_path,
            )
            model = model.merge_and_unload()

        if peft_model_name_or_path is not None:
            model = PeftModel.from_pretrained(
                model,
                peft_model_name_or_path,
            )
            if merge_peft:
                model = model.merge_and_unload()
        if extra_model_name_or_path is not None:
            logger.info(f"Loading extra model from {extra_model_name_or_path}")
            if not merge_peft:
                model = model.merge_and_unload()
            if isinstance(extra_model_name_or_path, str):
                model = PeftModel.from_pretrained(
                    model,
                    extra_model_name_or_path,
                )
                model = model.merge_and_unload()
            elif isinstance(extra_model_name_or_path, list):
                for extra_model in extra_model_name_or_path:
                    model = PeftModel.from_pretrained(
                        model,
                        extra_model,
                    )
                    peft_model_name_or_path = extra_model
                    model = model.merge_and_unload()
            else:
                raise ValueError(
                    f"extra_model_name_or_path should be a string or a list of strings."
                )
        config = {}
        config_addr = (
            peft_model_name_or_path
            if peft_model_name_or_path is not None
            else base_model_name_or_path
        )
        if os.path.exists(f"{config_addr}/llm2vec_config.json"):
            with open(f"{config_addr}/llm2vec_config.json", "r") as fIn:
                llm2vec_config = json.load(fIn)
            config.update(llm2vec_config)

        for key, value in encoder_args.items():
            config[key] = value

        llm2vec_model = cls(model=model, tokenizer=tokenizer, **config)
        
        # Auto-load latent attention weights if using latent_attention pooling
        if (hasattr(llm2vec_model, 'latent_attn') and 
            llm2vec_model.latent_attn is not None and 
            llm2vec_model.pooling_mode == "latent_attention"):
            
            llm2vec_model._load_latent_attention_weights(base_model_name_or_path, kwargs.get('use_safetensors', True))
        
        # Ensure the entire model is converted to the requested dtype
        if 'torch_dtype' in kwargs and kwargs['torch_dtype'] is not None:
            llm2vec_model = llm2vec_model.to(kwargs['torch_dtype'])
        
        return llm2vec_model