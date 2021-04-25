import os
import re

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm.autonotebook import tqdm
from train import CFG, TweetDataset, TweetModel, get_sentence


def load_trained_model(path, model_config, device):
    model = TweetModel(CFG.ROBERTA_PATH, model_config)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(CFG.TRAINED_MODEL_PATH, path)))
    model.eval()
    return model


def get_sentences_from_loader(data_loader, models, device):
    final_output = []
    start_char_indexes = []
    best_preds = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            orig_tweet = d["orig_tweet"]
            offsets = d["offsets"].numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            _start_preds = []
            _end_preds = []
            for model in models:
                _start_logits, _end_logits = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                _start_preds.append(torch.softmax(_start_logits, dim=1).cpu().detach().numpy())
                _end_preds.append(torch.softmax(_end_logits, dim=1).cpu().detach().numpy())

            start_preds = np.mean(_start_preds, axis=0)
            end_preds = np.mean(_end_preds, axis=0)

            for px, tweet in enumerate(orig_tweet):
                filtered_output, idx_start, idx_end, best_pred = get_sentence(
                    tweet=tweet, start_preds=start_preds[px, :], end_preds=end_preds[px, :], offsets=offsets[px]
                )
                start_char_indexes.append(offsets[px][idx_start][0])
                final_output.append(filtered_output)
                best_preds.append(best_pred)
    return final_output, start_char_indexes, best_preds


def adjust_predict(d):
    text = d["text"]
    predict = d["predict"]
    start_char_idx = d["start_char_index"]
    if not text.startswith(" "):
        return predict
    if d["best_pred"] >= 0.85:
        return predict
    is_last_char_punc = re.search(r'\w+[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]$', predict)
    if len(predict.split()) == 1 and is_last_char_punc:
        s = text[start_char_idx : (len(predict) + start_char_idx)]
    else:
        s = text[start_char_idx : (len(predict) + start_char_idx + 1)]
    return s


def pp_exclamation(x):
    s = x["predict"]
    p = x["best_pred"]
    if p >= 0.85:
        return s
    if len(s.split()) > 1:
        return s
    exist = re.search("\w+!{3,}$", s)
    if not exist:
        return s
    word = re.sub("!{3,}$", "", s)
    return f"{word}!!"


def pp_dot(x):
    s = x["predict"]
    p = x["best_pred"]
    if p >= 0.85:
        return s
    if len(s.split()) > 1:
        return s
    exist = re.search("\w+\.{3,}$", s)
    if not exist:
        return s
    word = re.sub("\.{3,}$", "", s)
    return f"{word}.."


def run():
    # load model
    device = torch.device("cuda")
    model_config = transformers.RobertaConfig.from_pretrained(CFG.ROBERTA_PATH)
    model_config.output_hidden_states = True

    models = []

    for i in range(5):
        model = load_trained_model(f"/model_{i}.bin", model_config, device)
        models.append(model)

    # data loader
    df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
    df_test.loc[:, "selected_text"] = df_test.text.values
    test_dataset = TweetDataset(
        tweet=df_test.text.values,
        sentiment=df_test.sentiment.values,
        selected_text=df_test.selected_text.values,
        tokenizer=CFG.TOKENIZER,
        max_len=CFG.MAX_LEN,
    )
    data_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=CFG.VALID_BATCH_SIZE)

    # predict
    final_output, start_char_indexes, best_preds = get_sentences_from_loader(data_loader, models, device)
    df_test.loc[:, "predict"] = final_output
    df_test.loc[:, "start_char_index"] = start_char_indexes
    df_test.loc[:, "best_pred"] = best_preds

    # post process
    df_test.loc[:, "predict"] = df_test.apply(lambda x: adjust_predict(x), axis=1)
    df_test.loc[:, "predict"] = df_test.apply(lambda x: pp_exclamation(x), axis=1)
    df_test.loc[:, "predict"] = df_test.apply(lambda x: pp_dot(x), axis=1)
    sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
    sample.loc[:, "selected_text"] = df_test["predict"]
    sample.to_csv("submission.csv", index=False)
