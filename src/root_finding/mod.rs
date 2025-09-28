// common helpers 
pub mod algorithms; 
pub mod report; 
pub mod errors; 
pub(crate) mod config;
pub(crate) mod signs; 
pub(crate) mod tolerances; 

// algorithms 
pub mod bisection;
pub mod regula_falsi;
pub mod secant;
pub mod newton;
pub mod brent; 
