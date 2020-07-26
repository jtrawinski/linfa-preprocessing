# linfa-preprocessing

`linfa-preprocessing` provides machine learning specific data preprocessing algorithms in the vein of Python's [sklearn.preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html).

Once this project matures, it will become part of the [`linfa`](https://github.com/rust-ml/linfa) Rust machine learning toolkit.

## Example

```rust
use ndarray::array;
use linfa_preprocessing::transformers::{Transformer, StandardScaler};
use approx::assert_abs_diff_eq;

let data = array![[2., 0.],
                  [0., 2.]];
let standardized = StandardScaler::fit_transform(&data);

assert_abs_diff_eq!(standardized, array![[0.707107, -0.707107], 
                                         [-0.707107, 0.707107]], 
                                         epsilon=1e-5);
```