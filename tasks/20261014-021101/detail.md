Computer physical memory is a flat, contiguous 1D array of bytes. A tensor represents an $n$-dimensional mathematical object defined by:
- **Shape Vector**: $S = (d_0, d_1, \dots, d_{n-1})$, where $d_i$ represents the size of dimension $i$.
- **Stride Vector**: $T = (s_0, s_1, \dots, s_{n-1})$, where $s_i$ indicates the number of scalar elements in storage that must be skipped to advance by 1 in dimension $i$.
- **Storage Offset**: An integer offset $\delta \ge 0$ denoting the starting index in the underlying storage buffer.

For any coordinate tuple $(i_0, i_1, \dots, i_{n-1})$ where $0 \le i_k < d_k$, the exact physical index in the 1D storage array is given by the linear affine transformation:
$$\text{LinearIndex}(i_0, i_1, \dots, i_{n-1}) = \delta + \sum_{k=0}^{n-1} i_k \cdot s_k$$

#### 3.1.2 Row-Major (C-Contiguous) Canonical Strides
In standard row-major format, the last dimension changes fastest in memory:
$$s_{n-1} = 1$$
$$s_k = s_{k+1} \cdot d_{k+1} = \prod_{j=k+1}^{n-1} d_j \quad \text{for } k \in [0, n-2]$$
The total number of logical elements is:
$$N = \prod_{k=0}^{n-1} d_k$$

