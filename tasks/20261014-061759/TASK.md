# Compaction of tensors

- STATUS: OPEN
- PRIORITY: 100
- TAGS: core,tensor

A tensor is contiguous if and only if its elements are laid out sequentially in memory without gaps or inversions.
Non-contiguous tensors occur after transpositions, non-uniform slices, or broadcasting. Hardware acceleration libraries
(cuBLAS, MPS, custom compute shaders) require contiguous memory blocks.

A contiguous() compaction step allocates a new buffer of length NN and
iterates through all logical coordinates in canonical order, copying values
from the strided source into the contiguous target.
