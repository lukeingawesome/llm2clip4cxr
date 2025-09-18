# training/zeroshot_retrieval.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

from eva_clip import get_cast_dtype
from .precision import get_autocast


def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive.clamp(min=1))
    return recall_at_k


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def evaluate(model, dataloader, tokenizer, device, precision, distributed=False, recall_k_list=[1, 5, 10]):
    """
    NOTE: This function uses encode_* APIs; you can ignore it if you're using the new retrieval_global below.
    Left here for compatibility with other callers.
    """
    batch_images_emb_list = []
    batch_texts_emb_list = []
    texts_image_index = []
    num_batches = dataloader.num_batches
    dataloader = dataloader_with_indices(dataloader)
    autocast = get_autocast(precision)
    cast_dtype = get_cast_dtype(precision)
    pbar = tqdm(total=num_batches)
    for batch_images, batch_texts, inds in dataloader:
        batch_images = batch_images.to(device, dtype=cast_dtype)
        # tokenize all texts in the batch
        if tokenizer:
            batch_texts_tok = tokenizer([text for _, texts in enumerate(batch_texts) for text in texts]).to(device)
        else:
            batch_texts_tok = torch.tensor([text for _, texts in enumerate(batch_texts) for text in texts]).to(device, dtype=cast_dtype)
        # store the index of image for each text
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for _ in texts]
        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            if distributed:
                batch_images_emb = F.normalize(model.module.encode_image(batch_images), dim=-1)
                batch_texts_emb = F.normalize(model.module.encode_text(batch_texts_tok), dim=-1)
            else:
                batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1)
                batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok), dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)

        pbar.update(1)

    batch_size = len(batch_images_emb_list[0])
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)
    try:
        scores = texts_emb @ images_emb.t()
    except Exception:
        scores = texts_emb.float() @ images_emb.t().float()

    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k) > 0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k) > 0).float().mean().item()

    return metrics


def retrieval_global(model,
                     text_encoder,  # raw LLM2Vec (no projection)
                     dataloader,
                     device,
                     precision,
                     distributed=False):
    """
    Simple diagonal retrieval (1 text per image in order) used by your OpenI/CheXpert evals.
    """
    autocast = get_autocast(precision)
    cast_dtype = get_cast_dtype(precision)
    vis_embeds_list, text_features_list = [], []

    for images, texts in tqdm(dataloader):
        images = images.to(device, dtype=cast_dtype)
        texts = texts.to(device)
        with torch.no_grad(), autocast():
            image_features = model.visual(images)
            text_features = model.text.projection(
                text_encoder(texts).to(dtype=cast_dtype)
            )
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

        vis_embeds_list.append(image_features)
        text_features_list.append(text_features)

    # concat after the loop
    image_features = torch.cat(vis_embeds_list,  dim=0)
    text_features  = torch.cat(text_features_list, dim=0)

    # Diagonal / rank@1
    i, correct, total = 0, 0, 0
    for i in range(text_features.shape[0]):
        sim = text_features[i] @ image_features.T
        correct_i = torch.argmax(sim)
        if i == correct_i:
            correct += 1
        total += 1
    t2i = correct / max(1, total)

    i, correct, total = 0, 0, 0
    for i in range(image_features.shape[0]):
        sim = image_features[i] @ text_features.T
        correct_i = torch.argmax(sim)
        if i == correct_i:
            correct += 1
        total += 1
    i2t = correct / max(1, total)

    metrics = {}
    metrics[f"image_retrieval_recall@1"] = t2i
    metrics[f"text_retrieval_recall@1"] = i2t

    def compute_recall(features1, features2, k_values):
        recalls = {}
        for k in k_values:
            correct = 0
            total = 0
            for j in range(features1.shape[0]):
                sim = features1[j] @ features2.T
                _, topk_indices = torch.topk(sim, k)
                if j in topk_indices:
                    correct += 1
                total += 1
            recalls[f"recall@{k}"] = correct / max(1, total)
        return recalls

    k_values = [1, 5, 10, 25, 50]
    t2i_recalls = compute_recall(text_features, image_features, k_values)
    for k, recall in t2i_recalls.items():
        metrics[f"image_retrieval_{k}"] = recall
    i2t_recalls = compute_recall(image_features, text_features, k_values)
    for k, recall in i2t_recalls.items():
        metrics[f"text_retrieval_{k}"] = recall

    return metrics


def retrieval_eval(model, text_encoder, data, epoch, args):
    if args.zeroshot_frequency == 0:
        return {}
    logging.info('Starting zero-shot retrieval.')

    model.to(args.device)
    collect_results = {}

    if 'openi' in data:
        logging.info('Starting Openi.')
        results = retrieval_global(model, text_encoder,
                                   data['openi'].dataloader, args.device, args.precision)
        for key in results.keys():
            collect_results['openi/' + key] = results[key]
    if 'chexpertplus' in data:
        logging.info('Starting Chexpert.')
        results = retrieval_global(model, text_encoder,
                                   data['chexpertplus'].dataloader, args.device, args.precision)
        for key in results.keys():
            collect_results['chexpert/' + key] = results[key]
    logging.info('Finished zero-shot retrieval.')
    return collect_results
