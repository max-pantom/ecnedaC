# 0001: Platform-pinned runtimes and PyAV media I/O

The Intel Mac uses Python 3.11 with PyTorch/TorchAudio 2.2.2 because newer official macOS x86
wheels are unavailable. Linux GPU jobs use Python 3.12 with the matched 2.11.0 line. Shared code
stays within their common API surface. PyAV owns media decoding and fixture generation;
`torchvision.io.read_video` is intentionally excluded.

NumPy is constrained to the 1.26 line because the final Intel macOS PyTorch 2.2.2 wheel was
compiled against the NumPy 1.x ABI.
