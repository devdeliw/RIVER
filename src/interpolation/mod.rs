pub mod algorithms; 
pub mod config; 
pub mod errors; 
pub mod report; 
pub mod traits;
pub use traits::Interpolator;

pub mod linear; 
pub mod newton; 
pub mod spline;
