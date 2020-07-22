use ndarray::{Array2, ArrayBase, Data, Ix2};
pub trait Transformer {
    fn fit(obs: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Self;
    fn transform(&self, obs: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64>;
    fn fit_transform(obs: &ArrayBase<impl Data<Elem = f64>, Ix2>) -> Array2<f64>;
}
