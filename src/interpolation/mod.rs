pub mod algorithms; 
pub mod config; 
pub mod errors; 
pub mod report; 
pub mod traits;
pub use traits::Interpolator;

pub mod linear; 
pub mod newton; 
pub mod spline;

pub use spline::natural   as natural_spline; 
pub use spline::clamped   as clamped_spline; 
pub use spline::monotonic as monotonic_spline;
