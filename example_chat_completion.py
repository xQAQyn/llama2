# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog
from file_compare import compare_files

import json
import re

def find_first_number(string):
    match = re.search(r'\d+', string)
    if match:
        return match.group()
    else:
        return -1

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = []
    with open("./data/piqa_tests.txt","r") as datafile:
        for line in datafile:
            data = json.loads(line)
            dialog = {}
            dialog["role"] = "user"
            dialog["content"] = "question:" + data["goal"] + "?" \
                                "options:" \
                                "\n 0." + data["sol1"] + \
                                "\n 1." + data["sol2"] + \
                                "Please respond with the number of the correct option:"
            dialogs.append([
                # {"role": "system", "content": "Please respond with the number of the correct option (1 or 2) : "},
                dialog
            ])
    
    # with open("./temp/dialogs.txt","w") as dialogLog:
        # json.dump(dialogs, dialogLog, ensure_ascii=False, indent=4)


    with open("/mnt/sevenT/xyn/llama2_log/result.txt","w") as output, open("/mnt/sevenT/xyn/llama2_log/log.txt","w") as logfile:
        # for i in range(0, len(dialogs), 8):
        for i in range(0, len(dialogs), 8):
            dialog_slice = dialogs[i:min(i+8,len(dialogs))]

            if i % 800 == 400:
                logfile.write(f"{i} dialogs processed\n")
                print(f"\n{i} dialogs processed")
                # output.flush()
                # compare_files("/mnt/sevenT/xyn/llama2_log/result.txt", "./data/train-labels.lst") 

            results = generator.chat_completion(
                dialog_slice,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
    
            for index, (dialog, result) in enumerate(zip(dialogs, results)):
                answer = result['generation']['content'].replace('\n',' ')
                output.write(
                    f"{find_first_number(answer)}\n"
                )
                # for msg in dialog:
                    # output.write(f"{msg['role'].capitalize()}: {msg['content']}\n")
                # output.write(
                    # f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}\n\n"
                # )
                # output.write("\n==================================\n")
    # compare_files("/mnt/sevenT/xyn/llama2_log/result.txt", "./data/train-labels.lst")


if __name__ == "__main__":
    fire.Fire(main)
