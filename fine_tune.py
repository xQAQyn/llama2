import fire
from llama import Llama, Dialog, Tokenizer

def main():
    ckpt_dir:str = "./llama-2-7b-chat/"
    tokenizer_path:str = "tokenizer.model"
    max_seq_len:int = 1024
    max_batch_size:int = 8

    model = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    print("model loaded")

    tokenizer = Tokenizer(model_path=tokenizer_path)
    print("tokenizer loaded")

if __name__ == "__main__":
    fire.Fire(main)
