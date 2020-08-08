use ndarray::{ArrayBase, Axis, DataMut, DataOwned, Ix2, ScalarOperand};
use ndarray_stats::{MaybeNan, QuantileExt};
use num_traits::{Float, FromPrimitive};
pub trait Preprocess<A> {
    /// Standardizes features by subtracting the mean and dividing by the standard devation.
    fn standard_scale(self) -> Self;

    /// Transforms each feature into the range [0., 1.]. This is equivalent to custom_scale(0., 1.).
    fn min_max_scale(self) -> Self;

    /// Transforms each feature into the range [min, max].
    fn custom_scale(self, min: A, max: A) -> Self;

    /// Transforms values to 0 or 1 depending on the provided threshold value.
    fn binarize(self, threshold: A) -> Self;
}

impl<A, S> Preprocess<A> for ArrayBase<S, Ix2>
where
    S: DataOwned<Elem = A> + DataMut,
    A: Float + FromPrimitive + MaybeNan + ScalarOperand,
    A::NotNan: Ord,
{
    /// Standardizes features by subtracting the mean and dividing by the standard devation.
    ///
    /// For each feature x in the matrix, returns (x - m) / s, where m is the sample mean and s is the sample standard deviation.
    ///
    /// # NaN
    /// This function does not work well with NaNs. If you have a NaN in a column, that column will contain all NaNs after calling this function.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use linfa_preprocessing::Preprocess;
    /// use approx::assert_abs_diff_eq;
    ///
    /// let data = array![[1., 3., 2.], [0., 0., 1.], [2., 0., 3.]];
    /// let standard_scaled = data.standard_scale();
    /// let expected_scaled = array![
    ///     [0., 1.154700538379, 0.],
    ///     [-1., -0.57735026919, -1.],
    ///     [1., -0.57735026919, 1.],
    /// ];
    /// assert_abs_diff_eq!(standard_scaled, expected_scaled, epsilon = 1e-5)
    /// ```
    fn standard_scale(self) -> Self {
        let stds = self.std_axis(Axis(0), A::one());
        let means = self.mean_axis(Axis(0)).unwrap();
        (self - means) / stds
    }

    /// Transforms each feature into the range [0., 1.]. This is equivalent to custom_scale(0., 1.).
    ///
    /// For each feature x in the array, returns (x - x.min) / (x.max - x.min).
    ///
    /// # NaN
    /// This function does not work well with NaNs. If you have a NaN in a column, that column will contain all NaNs after calling this function.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use linfa_preprocessing::Preprocess;
    ///
    /// let data = array![[-1., 2.], [-0.5, 6.], [0., 10.], [1., 18.]];
    /// let min_max_scaled = data.min_max_scale();
    /// let expected_scaled = array![[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]];
    /// assert_eq!(min_max_scaled, expected_scaled);
    /// ```
    fn min_max_scale(self) -> Self {
        let min = self.map_axis(Axis(0), |col| col.min_skipnan().clone());
        let max = self.map_axis(Axis(0), |col| col.max_skipnan().clone());
        (self - &min) / (max - min)
    }

    /// Transforms each feature into the range [min, max].
    ///
    /// For each feature x in the array, returns ((x - x.min) / (x.max - x.min)) * (max - min) + min.
    ///
    /// # NaN
    /// This function does not work well with NaNs. If you have a NaN in a column, that column will contain all NaNs after calling this function.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use linfa_preprocessing::Preprocess;
    ///
    /// let data = array![[-1., 2.], [-0.5, 6.], [0., 10.], [1., 18.]];
    /// let custom_scaled = data.custom_scale(-3., 5.);
    /// let expected_scaled = array![[-3., -3.], [-1., -1.], [1., 1.], [5., 5.]];
    /// assert_eq!(custom_scaled, expected_scaled);
    /// ```
    fn custom_scale(self, min: A, max: A) -> Self {
        let data_min = self.map_axis(Axis(0), |col| col.min_skipnan().clone());
        let data_max = self.map_axis(Axis(0), |col| col.max_skipnan().clone());
        ((self - &data_min) / (data_max - &data_min)) * (max - min) + min
    }

    /// Transforms values to 0 or 1 depending on the provided threshold value.
    ///
    /// For each value x in the array, if x < threshold then x = 0, else x = 1
    ///
    /// # NaN
    /// NaN will always compare as greater than. In other words, NaN will result in 1 regardless of the value of threshold.
    ///
    /// # Example
    /// ```
    /// use ndarray::array;
    /// use linfa_preprocessing::Preprocess;
    ///
    /// let data = array![[-1., 2.], [-0.5, 6.], [0., 10.], [1., 18.]];
    /// let binarized = data.binarize(0.);
    /// let expected_binarized = array![[0., 1.], [0., 1.], [1., 1.], [1., 1.]];
    /// assert_eq!(binarized, expected_binarized);
    /// ```
    fn binarize(self, threshold: A) -> Self {
        self.mapv_into(|val| if val < threshold { A::zero() } else { A::one() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    mod standard_scale_tests {
        use super::*;
        #[test]
        fn standard_scale() {
            let data = array![[1., 3., 2.], [0., 0., 1.], [2., 0., 3.]];
            let standard_scaled = data.standard_scale();
            let expected_scaled = array![
                [0., 1.154700538379, 0.],
                [-1., -0.57735026919, -1.],
                [1., -0.57735026919, 1.],
            ];
            assert_abs_diff_eq!(standard_scaled, expected_scaled, epsilon = 1e-5)
        }
    }

    mod min_max_tests {
        use super::*;
        #[test]
        fn min_max_scale() {
            let data = array![[-1., 2.], [-0.5, 6.], [0., 10.], [1., 18.]];
            let min_max_scaled = data.min_max_scale();
            let expected_scaled = array![[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]];
            assert_eq!(min_max_scaled, expected_scaled);
        }
    }

    mod custom_scaler_tests {
        use super::*;
        #[test]
        fn custom_scale() {
            let data = array![[-1., 2.], [-0.5, 6.], [0., 10.], [1., 18.]];
            let custom_scaled = data.custom_scale(-3., 5.);
            let expected_scaled = array![[-3., -3.], [-1., -1.], [1., 1.], [5., 5.]];
            assert_eq!(custom_scaled, expected_scaled);
        }
    }

    mod binarize_tests {
        use super::*;
        #[test]
        fn binarize() {
            let data = array![[-1., 2.], [-0.5, 6.], [0., 10.], [1., 18.]];
            let binarized = data.binarize(0.);
            let expected_binarized = array![[0., 1.], [0., 1.], [1., 1.], [1., 1.]];
            assert_eq!(binarized, expected_binarized);
        }
        #[test]
        fn binarize_nan() {
            let data = array![[-1., f64::NAN], [f64::NAN, 6.], [0., 10.], [1., 18.]];
            let binarized = data.binarize(10.);
            let expected_binarized = array![[0., 1.], [1., 0.], [0., 1.], [0., 1.]];
            assert_eq!(binarized, expected_binarized);
        }
    }

    mod compose_tests {
        use super::*;
        #[test]
        fn composable_min_std() {
            let data = array![[-1., 2.], [-0.5, 6.], [0., 10.], [1., 18.]];
            let min_max_std_scaled = data.min_max_scale().standard_scale();
            let expected_scaled = array![
                [-1.024695, -1.024695],
                [-0.439155, -0.439155],
                [0.146385, 0.146385],
                [1.317465, 1.317465]
            ];
            assert_abs_diff_eq!(min_max_std_scaled, expected_scaled, epsilon = 1e-5);
        }

        #[test]
        fn composable_min_std_bin() {
            let data = array![[-1., 2.], [-0.5, 6.], [0., 10.], [1., 18.]];
            let min_max_std_scaled = data.min_max_scale().standard_scale().binarize(0.);
            let expected_scaled = array![[0., 0.], [0., 0.], [1., 1.], [1., 1.]];
            assert_eq!(min_max_std_scaled, expected_scaled);
        }

        #[test]
        fn readme_example() {
            let data = array![[-1., 2.], [-0.5, 6.], [0., 10.], [1., 18.]];

            let processed = data.min_max_scale().standard_scale().binarize(0.);

            let expected = array![[0., 0.], [0., 0.], [1., 1.], [1., 1.]];

            assert_eq!(processed, expected)
        }
    }
}
