use rustfft::FftNum;

// A trait for sample types, which includes f32 and f64. 
// This is used to generalize the code for different sample types.
pub trait Sample:
    FftNum + 
    num::Float + 
    num::FromPrimitive + 
    rand::distr::uniform::SampleUniform +
    Clone + 
    Copy + 
    Default + 
    'static
{
}

impl Sample for f32 {}
impl Sample for f64 {}
