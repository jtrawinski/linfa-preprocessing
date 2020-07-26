pub use super::transformer::Transformer;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use std::fmt;

#[derive(Clone, PartialEq, Default)]
/// StandardScalar standardizes features by subtracting the mean and dividing by the sample standard deviation.
/// This results in features with zero mean and unit variance.
// TODO: Allow computation without mean (just scale) or without stddev (just center)
// TODO: Allow for online computation (partial_fit)
pub struct StandardScaler {
    means: Array1<f64>,
    stds: Array1<f64>,
}

impl StandardScaler {
    pub fn means(&self) -> &Array1<f64> {
        &self.means
    }

    pub fn stds(&self) -> &Array1<f64> {
        &self.stds
    }
}

impl Transformer for StandardScaler {
    /// Returns a `StandardScalar` instance with means and standard deviations derived from `obs`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use linfa_preprocessing::transformers::{Transformer, StandardScaler};
    /// use approx::assert_abs_diff_eq;
    ///
    /// let data = array![[2., 0.], [0., 2.]];
    /// let std_sclr = StandardScaler::fit(&data);
    ///
    /// assert_eq!(*std_sclr.means(), array![1., 1.]);
    /// // std dev is sqrt(2) for each column
    /// assert_abs_diff_eq!(*std_sclr.stds(), array![1.41421356237, 1.41421356237], epsilon=1e-5);
    /// ```
    ///
    /// # Panics
    /// This function panics if a column is constant.
    /// This is because the column will have a standard deviation of zero, which results in a NaN when transforming due to division by zero.
    fn fit(obs: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> StandardScaler {
        // Using sample standard deviation (ddof = 1)
        let stds = obs.std_axis(Axis(0), 1.);
        if stds.iter().any(|std| *std == 0.) {
            // TODO: Tell user which column(s) have stddev of zero.
            // Should this panic or deal with the error in another way?
            panic!("A column has a standard deviation of zero. Cannot standardize due to divison by zero.");
        }
        StandardScaler {
            means: obs.mean_axis(Axis(0)).unwrap(),
            stds,
        }
    }

    /// Uses the means and standard deviations in `self` to standardize the features of obs.
    ///
    /// Returns a new owned Array.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use linfa_preprocessing::transformers::{Transformer, StandardScaler};
    /// use approx::assert_abs_diff_eq;
    ///
    /// let data = array![[2., 0.], [0., 2.]];
    /// let std_sclr = StandardScaler::fit(&data);
    /// let standardized = std_sclr.transform(&data);
    ///
    /// assert_abs_diff_eq!(standardized, array![[0.707107, -0.707107], [-0.707107, 0.707107]], epsilon=1e-5);
    /// ```
    ///
    /// Standardization is calculated as z = (x - m) / s where z is the resulting standardized feature,
    /// m is the original feature mean and s is the original feature sample standard deviation.
    fn transform(&self, obs: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        (obs - &self.means) / &self.stds
    }

    /// Applies `fit` and then `transform` in succession.
    ///
    /// Returns a new owned Array.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use linfa_preprocessing::transformers::{Transformer, StandardScaler};
    /// use approx::assert_abs_diff_eq;
    ///
    /// let data = array![[2., 0.], [0., 2.]];
    /// let standardized = StandardScaler::fit_transform(&data);
    ///
    /// assert_abs_diff_eq!(standardized, array![[0.707107, -0.707107], [-0.707107, 0.707107]], epsilon=1e-5);
    /// ```
    ///
    /// # Panics
    /// This function panics if a column is constant. This results in a standard deviation of zero, which results in a NaN when transforming due to division by zero.
    fn fit_transform(obs: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        Self::fit(obs).transform(obs)
    }
}

impl fmt::Display for StandardScaler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Means: {}\nStds: {}", self.means, self.stds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    #[test]
    fn fit_test() {
        let data = array![[1., 3., 2.], [0., 0., 1.], [2., 0., 3.]];
        let standard_scaler = StandardScaler::fit(&data);
        let expected_means = array![1., 1., 2.];
        let expected_stds = array![1.0, 1.732050807569, 1.];
        assert_abs_diff_eq!(standard_scaler.means, expected_means, epsilon = 1e-5);
        assert_abs_diff_eq!(standard_scaler.stds, expected_stds, epsilon = 1e-5);
    }

    #[test]
    fn transform_test() {
        let data = array![[1., 3., 2.], [0., 0., 1.], [2., 0., 3.]];
        let standard_scaler = StandardScaler::fit(&data);
        let standard_scaled = standard_scaler.transform(&data);
        let expected_scaled = array![
            [0., 1.154700538379, 0.],
            [-1., -0.57735026919, -1.],
            [1., -0.57735026919, 1.]
        ];
        assert_abs_diff_eq!(standard_scaled, expected_scaled, epsilon = 1e-5);
    }

    #[test]
    fn fit_transform_test() {
        let data = array![[1., 3., 2.], [0., 0., 1.], [2., 0., 3.]];
        let standard_scaled = StandardScaler::fit_transform(&data);
        let expected_scaled = array![
            [0., 1.154700538379, 0.],
            [-1., -0.57735026919, -1.],
            [1., -0.57735026919, 1.]
        ];
        assert_abs_diff_eq!(standard_scaled, expected_scaled, epsilon = 1e-5);
    }

    #[test]
    fn empty_array_test() {
        let data: Array2<f64> = array![[]];
        let standard_scaled = StandardScaler::fit_transform(&data);
        assert_eq!(standard_scaled, array![[]]);
    }

    #[test]
    #[should_panic(
        expected = "A column has a standard deviation of zero. Cannot standardize due to divison by zero."
    )]
    fn zero_stddev_fit() {
        // third column is constant
        let data = array![[1., 1., 1.], [2., 3., 1.,]];
        StandardScaler::fit(&data);
    }

    #[test]
    #[should_panic(
        expected = "A column has a standard deviation of zero. Cannot standardize due to divison by zero."
    )]
    fn zero_stddev_fit_transform() {
        // third column is constant
        let data = array![[1., 1., 1.], [2., 3., 1.,]];
        StandardScaler::fit_transform(&data);
    }
}
