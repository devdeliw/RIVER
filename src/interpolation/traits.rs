use crate::interpolation::errors::InterpolationError;

pub trait Interpolator {
    /// evaluates single point
    /// defined separately in each method
    fn eval(&self, x: f64) -> Result<f64, InterpolationError>;

    /// evaluates many points
    #[inline]
    fn eval_many(&self, xs: &[f64]) -> Result<Vec<f64>, InterpolationError> {
        xs.iter().map(|&xq| self.eval(xq)).collect()
    }
}

