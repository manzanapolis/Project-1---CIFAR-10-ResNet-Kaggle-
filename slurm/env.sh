# Python environment on NYU HPC Cloud Bursting: NYU's prebuilt PyTorch 2.9.1 / CUDA 13 container
# (torch 2.9.1, torchvision 0.24.1, numpy 2.3). The wrapper adds --nv automatically on GPU nodes.
PYTORCH_RUN=/share/apps/pytorch/2.9.1/run-pytorch.bash

# py ARGS... : run python inside the container
py() {
  "$PYTORCH_RUN" python "$@"
}
