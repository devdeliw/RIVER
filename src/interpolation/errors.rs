use thiserror::Error;

#[derive(Debug, Error)]
pub enum InterpolationError {
    #[error("unequal length: x has {x_len} elements, y has {y_len}")]
    UnequalLength { x_len: usize, y_len: usize },

    #[error("non-finite value in input vector at index {idx}")]
    NonFiniteVec { idx: usize },

    #[error("empty input vector(s)")]
    EmptyInput,

    #[error("insufficient points: got {got}, need at least 2")]
    InsufficientPoints { got: usize },

    #[error("duplicate x-values detected: {x1} and {x2}")]
    DuplicateX { x1: f64, x2: f64 },

    #[error("x-values must be strictly increasing")]
    NonIncreasingX,

    #[error("evaluation point {got} out of bounds in ({x_min}, {x_max})")]
    OutOfBounds { got: f64, x_min: f64, x_max: f64},

    #[error("invalid x_tol {got} must be finite and > 0")]
    InvalidXTol { got: f64 } 
}

