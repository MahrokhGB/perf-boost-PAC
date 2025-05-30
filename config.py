import torch, os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, BASE_DIR)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('[INFO] GPU available')
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print('[INFO] running on Apple silicon GPU')
else:
    if not torch.backends.mps.is_built():
        print("[INFO] MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("[INFO] MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    device = torch.device("cpu")
    print('[INFO] running on CPU')

