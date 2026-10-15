# Support 0-sized dimension

- STATUS: OPEN
- PRIORITY: 100
- TAGS: core,dimension

Currently, there is no support for 0-sized dimensions. (shapes with 0 in it)
eg: creating a tensor with shape [2,0,3] will result in runtime error, 0 must NOT be in
shape.

Implement 0-sized dimensions, this is used in CNNs
This allows dynamic slicing (start..end when start == end)
When writing algorithms that slice windows dynamically:
    If start == end (e.g., 5..5):
    slice = tensor[5:5] Yields shape [0, 128]
Without 0-sized dimensions, slicing tensor[5:5] would crash your program instead of gracefully returning an empty slice.


