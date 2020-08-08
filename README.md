# linfa-preprocessing

`linfa-preprocessing` provides machine learning specific data preprocessing algorithms in the vein of Python's [`sklearn.preprocessing`](https://scikit-learn.org/stable/modules/preprocessing.html).

Once this project matures, it will become part of the [`linfa`](https://github.com/rust-ml/linfa) Rust machine learning toolkit.

## Functionality

`linfa-preprocessing` consists of a single trait: `Preprocess<A>`. All of the preprocessing functions we provide are defined in `Preprocessing<A>`. It is generic over a single parameter `A`, representing the element type of the data being operated on.

There is one implementation of this trait that is compatible with 2-dimensional, float-based, owned-memory `ndarray` arrays. In this implementation, the preprocessing functions consume the array and produce a new owned array. This allows for ergonomic chaining of preprocessing methods.

### Functions

- [x] StandardScale
- [x] MinMaxScale
- [x] CustomScale
- [x] Binarize
- [ ] RobustScale
- [ ] Normalize
- [ ] PowerTransform
- [ ] QuantileTransform
- [ ] FunctionTransform
- [ ] LabelBinarize
- [ ] MultiLabelBinarize
- [ ] PolynomialFeatures
- [ ] LabelEncode

## Example

```rust
use linfa_preprocessing::Preprocess;
use ndarray::array;

let data = array![[-1., 2.],
                  [-0.5, 6.],
                  [0., 10.],
                  [1., 18.]];

let processed = data.min_max_scale()
                    .standard_scale()
                    .binarize(0.);

let expected = array![[0., 0.],
                      [0., 0.],
                      [1., 1.],
                      [1., 1.]];

assert_eq!(processed, expected)
```
