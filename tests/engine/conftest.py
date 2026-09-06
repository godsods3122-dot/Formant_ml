import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import torch
torch.set_num_threads(2)   # 이 컨테이너에서 4 스레드는 원소별 연산이 1000 배 느리다
