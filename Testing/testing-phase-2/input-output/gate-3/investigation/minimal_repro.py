"""Five original inverse failures, directly in installed gReLU; no Studio code."""
import torch
from grelu.sequence.format import check_one_hot, one_hot_to_strings

for base, vector in zip('ACGTN', [[1, 0, 0, 0], [0, 1, 0, 0],
                                [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]):
    tensor = torch.tensor(vector, dtype=torch.float32).reshape(1, 4, 1)
    try:
        actual = one_hot_to_strings(tensor)
    except Exception as error:
        actual = f'{type(error).__name__}: {error}'
    print(f'{base}: shape={list(tensor.shape)} accepted={check_one_hot(tensor)} '
          f'expected={[base]!r} actual={actual!r}', flush=True)
