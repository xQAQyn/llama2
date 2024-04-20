import torch

# 加载 checkpoint 文件
checkpoint = torch.load('/mnt/sevenT/xyn/consolidated.01.pth', map_location='cpu')

for key in checkpoint.keys():
    print(f"{key}: {checkpoint[key].dtype}")

# with open("/mnt/sevenT/xyn/llama2_log/ckpt_log") as log:
# 遍历模型的状态字典并打印每个参数的数据类型
    # for name, param in model_state_dict.items():
        # log.write(f'{name} - {param.dtype}\n')

# is_half_precision = any(p.dtype == torch.float16 for p in model_state_dict.values())
# print('模型使用半精度:', is_half_precision)
