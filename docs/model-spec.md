# Encoder model specification

The video encoder is a compact R(2+1)D network. The audio encoder is a compact residual CNN over
log-mel spectrograms. Both return timestamped, masked temporal tokens and normalized global
projection embeddings. GPU research defaults are approximately 14.59M video parameters and
11.32M audio parameters; local profiles use the same code at reduced width.

