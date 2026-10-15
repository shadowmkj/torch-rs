# Tensor Creation & Zero-Copy View Methods

- STATUS: OPEN
- PRIORITY: 100
- TAGS: core,tensor

## Implement zero copy methods for Tensor
- Implement constructors:
  - `zeros(shape, device)`
  - `ones(shape, device)`
  - `from_slice(slice, shape, device)`
  - `randn(shape, mean, std, device)`
- Implement zero-copy transformations:
  - `transpose(dim0, dim1)`: Swaps shape and stride entries; returns a new `Tensor` pointing to the exact same storage.
  - `reshape(new_shape)`: If contiguous, computes new strides directly; if non-contiguous, triggers compaction first.
  - `slice(dim, start, end)`: Adjusts offset and dimension size.
  - `broadcast_to(target_shape)`: Returns a new tensor handle with zero-strides on expanded dimensions.
