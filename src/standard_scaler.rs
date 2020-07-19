use ndarray::{Data, Array1, Array2, ArrayBase, Ix2, Axis};
struct StandardScaler {
    means: Option<Array1<f64>>,
    stds: Option<Array1<f64>>,
    n_samples_seen: usize,
}

struct StandardScalerParams {
    with_means: bool,
    with_stds: bool,
}

impl StandardScaler {
    fn fit(observations: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> StandardScaler {
        StandardScaler {
            means: observations.mean_axis(Axis(0)),
            stds: Some(observations.std_axis(Axis(0), 1.)),
            n_samples_seen: observations.nrows(),
        }
    }

    fn fit_with(observations: &ArrayBase<impl Data<Elem = f64>, Ix2>, params: StandardScalerParams) -> StandardScaler {
        StandardScaler {
            means: if params.with_means { observations.mean_axis(Axis(1)) } else { None },
            stds: if params.with_stds { Some(observations.std_axis(Axis(1), 1.)) } else { None },
            n_samples_seen: observations.nrows(),
        }
    }

    fn center(&self, observations: &ArrayBase<impl Data<Elem = f64>, Ix2>, axis: Axis) -> Array2<f64> {
       observations - self.means.as_ref().unwrap()
    }

    fn scale(&self, observations: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        observations / self.stds.as_ref().unwrap()
    }

    fn transform(&self, observations: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64> {
        (observations - self.means.as_ref().unwrap()) / self.stds.as_ref().unwrap()
    }

    fn transform_inplace(&self, observations: ArrayBase<impl Data<Elem = f64>, Ix2>) {

    }
}