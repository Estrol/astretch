use num::Complex;

pub trait ComplexTrait<T: super::Sample> {
    fn real(&self) -> T;
    fn imag(&self) -> T;
}

impl<T: super::Sample> ComplexTrait<T> for Complex<T> {
    #[inline(always)]
    fn real(&self) -> T {
        self.re
    }

    #[inline(always)]
    fn imag(&self) -> T {
        self.im
    }
}
