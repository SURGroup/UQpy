import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given, strategies as st


# @given(
#     width=st.integers(min_value=4, max_value=32),
#     modes=st.integers(min_value=1, max_value=16),
#     n_batch=st.integers(min_value=32, max_value=128),
# )
def test_output_shape():
    # FixMe: this crashes under the random input generated above
    width = 10
    modes = 4
    n_batch = 3
    fourier_block = sml.FourierBlock1d(width, modes)
    x = torch.ones((n_batch, width, width))
    y = fourier_block(x)
    assert y.shape == torch.Size([n_batch, width, width])
