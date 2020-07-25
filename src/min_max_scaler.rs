use crate::transformer::Transformer;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_stats::QuantileExt;

/// Transforms each feature by scaling to the range [0, 1]

/// If x is the original feature, then the transformed feature z = (x - min(x)) / (max(x) - min(x))
// TODO: Allow for custom range scaling instead of just [0, 1]
// TODO: Allow for different NaN handling strategies.
pub struct MinMaxScaler {
    min: Array1<f64>,
    max: Array1<f64>,
}

impl MinMaxScaler {
    pub fn min(&self) -> &Array1<f64> {
        &self.min
    }
    pub fn max(&self) -> &Array1<f64> {
        &self.max
    }
}

impl Transformer for MinMaxScaler {
    /// Returns a `MinMaxScaler` instance with min and max derived from the columns of `obs`.
    ///
    /// # NaN
    /// This function ignores `NaN`s in the calculation of the min and max.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use linfa_preprocessing::min_max_scaler::MinMaxScaler;
    /// use linfa_preprocessing::transformer::Transformer;
    /// use approx::assert_abs_diff_eq;
    ///
    /// let data = array![[1., 3., 2.], [5., 2., 1.]];
    /// let mms = MinMaxScaler::fit(&data);
    /// assert_abs_diff_eq!(*mms.min(), array![1., 2., 1.]);
    /// assert_abs_diff_eq!(*mms.max(), array![5., 3., 2.]);
    /// ```
    fn fit(obs: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> MinMaxScaler {
        MinMaxScaler {
            min: obs.map_axis(Axis(0), |col| col.min_skipnan().clone()),
            max: obs.map_axis(Axis(0), |col| col.max_skipnan().clone()),
        }
    }

    /// Use the min and max from `self` to standardize the features of `obs`.
    ///
    /// Returns a new owned Array
    ///
    /// # NaN
    /// This function will pass through any NaNs from `obs`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use linfa_preprocessing::min_max_scaler::MinMaxScaler;
    /// use linfa_preprocessing::transformer::Transformer;
    ///
    /// let data = array![[-1., 2.], [-0.5, 6.], [0., 10.], [1., 18.]];
    /// let mms = MinMaxScaler::fit(&data);
    /// let mms_data = mms.transform(&data);
    /// assert_eq!(
    ///     mms_data,
    ///     array![[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]]
    /// );
    ///
    /// ```
    fn transform(&self, obs: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        (obs - self.min()) / (self.max() - self.min())
    }

    /// Applies `fit` and then `transform` in succession.
    ///
    /// Returns a new owned Array.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use linfa_preprocessing::min_max_scaler::MinMaxScaler;
    /// use linfa_preprocessing::transformer::Transformer;
    /// let data = array![[-1., 2.], [-0.5, 6.], [0., 10.], [1., 18.]];
    /// let mms_data = MinMaxScaler::fit_transform(&data);
    /// assert_eq!(
    ///     mms_data,
    ///     array![[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]]
    /// );
    /// ```
    fn fit_transform(obs: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        Self::fit(obs).transform(obs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    #[test]
    fn fit_test() {
        let data = array![[1., 3., 2.], [5., 2., 1.]];
        let mms = MinMaxScaler::fit(&data);
        assert_abs_diff_eq!(*mms.min(), array![1., 2., 1.]);
        assert_abs_diff_eq!(*mms.max(), array![5., 3., 2.]);
    }

    #[test]
    fn transform_test() {
        let data = array![[-1., 2.], [-0.5, 6.], [0., 10.], [1., 18.]];
        let mms = MinMaxScaler::fit(&data);
        let mms_data = mms.transform(&data);
        assert_eq!(
            mms_data,
            array![[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]]
        );
    }

    #[test]
    fn fit_transform_test() {
        let data = array![[-1., 2.], [-0.5, 6.], [0., 10.], [1., 18.]];
        let mms_data = MinMaxScaler::fit_transform(&data);
        assert_eq!(
            mms_data,
            array![[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]]
        );
    }

    #[test]
    fn nan_fit_test() {
        let data = array![[1., 3., 2.], [5., 2., 1.], [f64::NAN, f64::NAN, f64::NAN]];
        let mms = MinMaxScaler::fit(&data);
        assert_eq!(*mms.min(), array![1., 2., 1.]);
        assert_eq!(*mms.max(), array![5., 3., 2.]);
    }
}
