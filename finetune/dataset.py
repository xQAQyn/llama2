import copy
import datasets
import itertools

B_INST, E_INST = "[INST]", "[/INST]"

def tokenize_dialog(dialog, tokenizer):
    if tokenizer.vocab_size >= 128000:
        dialog_tokens = tokenizer.apply_chat_template(dialog)
        dialog_tokens = dialog_tokens[:-4] # Remove generation prompt <|start_header_id|>assistant<|end_header_id|>\n\n
        eot_indices = [i for i,n in enumerate(dialog_tokens) if n == 128009]
        labels = copy.copy(dialog_tokens)
        last_idx = 0
        for n, idx in enumerate(eot_indices):
            if n % 2 == 1:
                last_idx = idx
            else:
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)

        dialog_tokens = [dialog_tokens]
        labels_tokens = [labels]
    else:
        prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog[::2]]
        answer_tokens = [tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in dialog[1::2]]
        dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))

        #Add labels, convert prompt token to -100 in order to ignore in loss function
        labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))

def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = dataset.load_dataset("piqa", split=split)
    dataset = dataset.map(lambda sample: [{
        "role": "user",
        "content": (
            f"question: {sample['goal']}?\n"
            f"option 0: {sample['sol1']}\n"
            f"option 1: {sample['sol2']}\n"
            "Please respond with the number of the correct option:"
        )
    },{
        "role": "assistant",
        "content": sample["label"]
    }])
    dataset = dataset.map(lambda dialog: tokenize_dialog(dialog, tokenizer), batched=True)
    return dataset