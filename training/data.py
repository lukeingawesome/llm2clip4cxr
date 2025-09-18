import re
import random
from typing import Union, List, Tuple
from dataclasses import dataclass

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def shuffle_sentences(text: str, probability=0.5, sectioned: bool = False, seed: Union[int, None] = None) -> str:
    """
    Shuffle sentences in a text. If sectioned=True, only shuffle within
    [FINDINGS]{...} and [IMPRESSION]{...} bodies independently.

    Args:
        text: The full report text.
        sectioned: If True, shuffle sentences separately within the FINDINGS
                   and IMPRESSION sections.
        seed: Optional seed for deterministic shuffling.

    Returns:
        The text with sentences shuffled.
    """
    rng = random.Random(seed)

    if random.random() > probability:
        return text

    def _split_sentences(s: str) -> List[str]:
        # Split on sentence-ending punctuation followed by whitespace/newline.
        # Keeps the punctuation with the sentence.
        s = s.strip()
        if not s:
            return []
        return re.split(r'(?<=[.!?])\s+', s)

    def _shuffle_body(body: str) -> str:
        sentences = _split_sentences(body)
        if len(sentences) <= 1:
            return body.strip()
        rng.shuffle(sentences)
        return " ".join(sentences).strip()

    if not sectioned:
        return _shuffle_body(text)

    # --- Sectioned mode ---
    # Capture bodies; tolerant of extra spaces/newlines; case-insensitive.
    # FINDINGS: everything up to [IMPRESSION] or end of string
    m_find = re.search(r'(?is)\[\s*FINDINGS\s*\](.*?)(?=\[\s*IMPRESSION\s*\]|$)', text)
    m_impr  = re.search(r'(?is)\[\s*IMPRESSION\s*\](.*)$', text)

    if not (m_find or m_impr):
        # Fallback: no recognizable sections — shuffle the whole thing.
        return _shuffle_body(text)

    def _extract(body_raw: str) -> Tuple[str, bool]:
        body = body_raw.strip()
        has_braces = body.startswith("{") and body.endswith("}")
        if has_braces:
            body = body[1:-1].strip()
        return body, has_braces

    # Extract and shuffle each section if present
    findings_body, findings_has_braces = ("", False)
    impression_body, impression_has_braces = ("", False)

    if m_find:
        findings_body, findings_has_braces = _extract(m_find.group(1))
        findings_body = _shuffle_body(findings_body)

    if m_impr:
        impression_body, impression_has_braces = _extract(m_impr.group(1))
        impression_body = _shuffle_body(impression_body)

    # Rebuild output.
    # Preserve braces only if they were originally present for that section.
    parts = []
    # Preserve any leading text before [FINDINGS]
    lead = text[:m_find.start()] if m_find else text[:m_impr.start()]
    if lead.strip():
        parts.append(lead.strip())

    if m_find:
        fb = f"{{{findings_body}}}" if findings_has_braces else findings_body
        parts.append("[FINDINGS] " + fb)

    if m_impr:
        ib = f"{{{impression_body}}}" if impression_has_braces else impression_body
        parts.append("[IMPRESSION] " + ib)

    # Preserve any trailing text after [IMPRESSION]
    if m_impr:
        tail = text[m_impr.end():].strip()
        if tail:
            parts.append(tail)

    # Join with a single blank between blocks
    return re.sub(r'\s+', ' ', " ".join(p for p in parts if p)).strip()


class CustomCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None, img_key='image_path', caption_key='caption', tokenizer=None, is_train=True, separator='!@#$%^&*()'):
        """
        Args:
            csv_file (string): Path to the csv file
            transform (callable, optional): Optional transform to be applied on images
            img_key (string): Column name for image paths
            caption_key (string): Column name for captions
            tokenizer (callable, optional): Optional tokenizer for processing captions
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.img_key = img_key
        self.caption_key = caption_key
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.separator = separator
        
        # Check for required columns and add warnings/fake columns if missing
        self._validate_and_fix_columns()
        
        # Pre-compute instruction templates to avoid string operations in __getitem__
        self.base_instruction = 'Retrieve the image that best matches the following report.; '
        self.findings_instruction = 'Retrieve the image that best matches the following report for the findings section.; '
        self.impression_instruction = 'Retrieve the image that best matches the following report for the impression section.; '
        
        # Pre-compute separator + instruction combinations for automatic section detection
        self.findings_prefix = self.findings_instruction + self.separator
        self.impression_prefix = self.impression_instruction + self.separator
        self.base_prefix = self.base_instruction + self.separator
    
    def _validate_and_fix_columns(self):
        """Check for required columns and add warnings/fake columns if missing."""
        import warnings
        
        # Check if img_key column exists
        if self.img_key not in self.data_frame.columns:
            warnings.warn(
                f"Column '{self.img_key}' not found in CSV file. "
                f"Available columns: {list(self.data_frame.columns)}. "
                f"Adding fake column with placeholder values.",
                UserWarning
            )
            # Add fake column with placeholder values
            self.data_frame[self.img_key] = f"fake_image_path_{self.img_key}"
        
        # Check if caption_key column exists
        if self.caption_key not in self.data_frame.columns:
            warnings.warn(
                f"Column '{self.caption_key}' not found in CSV file. "
                f"Available columns: {list(self.data_frame.columns)}. "
                f"Adding fake column with placeholder values.",
                UserWarning
            )
            # Add fake column with placeholder values
            self.data_frame[self.caption_key] = f"fake_caption_{self.caption_key}"
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        """Returns one sample of data"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get image path and caption - use .iloc for faster access
        row = self.data_frame.iloc[idx]
        img_path = row[self.img_key]
        caption = str(row[self.caption_key])
        
        # Apply text augmentation during training
        if self.is_train:
            # Determine if sectioned mode should be used based on caption content
            sectioned = '[FINDINGS]' in caption or '[IMPRESSION]' in caption
            caption = shuffle_sentences(caption, probability=0.3, sectioned=sectioned)

        # Automatically detect section type and use appropriate instruction prefix
        if '[FINDINGS]' in caption:
            caption = self.findings_prefix + caption
        elif '[IMPRESSION]' in caption:
            caption = self.impression_prefix + caption
        else:
            caption = self.base_prefix + caption
        
        # Load and process image
        if img_path.startswith('fake_image_path_'):
            # Create an empty white image for fake paths
            image = Image.new('RGB', (224, 224), color='white')
        else:
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                # If image loading fails, create an empty white image
                import warnings
                warnings.warn(f"Failed to load image '{img_path}': {e}. Using empty image instead.", UserWarning)
                image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
            
        return image, caption
    def collate_fn(self, batch):
        images, texts = zip(*batch)
        images = torch.stack(images)
        
        # Split texts
        texts_2 = []
        original_texts = []
        for text in texts:
            t = text.split(self.separator)
            texts_2.append(t[1] if len(t) > 1 else "")
            original_texts.append("".join(t))

        # Tokenize original texts with padding
        
        original = self.tokenizer(
            original_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Process secondary texts and create embed masks
        embed_mask = torch.zeros_like(original["attention_mask"])
        for i, t in enumerate(texts_2):
            if t:  # Only process non-empty secondary texts
                ids = self.tokenizer(
                    [t],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                    add_special_tokens=False,
                )
                if len(ids["input_ids"][0]) > 0:
                    embed_mask[i, -len(ids["input_ids"][0]):] = 1

        original["embed_mask"] = embed_mask
        return images, original


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_cxr_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CustomCSVDataset(
        csv_file=input_filename,
        transform=preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        tokenizer=tokenizer,
        is_train=is_train)
    
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=dataset.collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_cxr_dataset(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_cxr_dataset(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    return data