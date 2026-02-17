use num::complex::Complex;

use crate::dsp::Sample;

#[inline(always)]
pub fn make_split_complex<T>(vec: &[Complex<T>]) -> (&[T], &[T])
where
    T: Sample,
{
    let len = vec.len();
    unsafe {
        let ptr = vec.as_ptr() as *const T;
        let re_slice = std::slice::from_raw_parts(ptr, len);
        let im_slice = std::slice::from_raw_parts(ptr.add(len), len);
        (re_slice, im_slice)
    }
}

#[inline(always)]
pub fn make_split_complex_mut<T>(vec: &mut [Complex<T>]) -> (&mut [T], &mut [T])
where
    T: Sample,
{
    let len = vec.len();
    unsafe {
        let ptr = vec.as_mut_ptr() as *mut T;
        let re_slice = std::slice::from_raw_parts_mut(ptr, len);
        let im_slice = std::slice::from_raw_parts_mut(ptr.add(len), len);
        (re_slice, im_slice)
    }
}

#[inline(always)]
pub fn complex_mul_single<T>(a: &mut [Complex<T>], c: &[Complex<T>], size: usize)
where
    T: Sample + core::ops::Mul<Output = T>,
{
    for i in 0..size {
        a[i] = a[i] * c[i];
    }
}

#[inline(always)]
pub fn complex_mul_conj_single<T>(a: &mut [Complex<T>], c: &[Complex<T>], size: usize)
where
    T: Sample + core::ops::Mul<Output = T>,
{
    for i in 0..size {
        a[i] = a[i] * c[i].conj();
    }
}

#[inline(always)]
pub fn complex_mul_split_single<T>(ar: &mut [T], ai: &mut [T], cr: &[T], ci: &[T], size: usize)
where
    T: Sample
        + core::ops::Mul<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Add<Output = T>,
{
    for i in 0..size {
        let br = ar[i];
        let bi = ai[i];
        ar[i] = br * cr[i] - bi * ci[i];
        ai[i] = br * ci[i] + bi * cr[i];
    }
}

#[inline(always)]
pub fn complex_mul_conj_split_single<T>(ar: &mut [T], ai: &mut [T], cr: &[T], ci: &[T], size: usize)
where
    T: Sample
        + core::ops::Mul<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Add<Output = T>,
{
    for i in 0..size {
        let br = ar[i];
        let bi = ai[i];
        ar[i] = br * cr[i] + bi * ci[i];
        ai[i] = bi * cr[i] - br * ci[i];
    }
}

#[inline(always)]
pub fn interleave_copy<const SIZE: usize, T>(a: &[T], b: &mut [T], b_stride: usize)
where
    T: Copy,
{
    for bi in 0..b_stride {
        let offset_a = bi * SIZE;
        let offset_b = bi;
        for ai in 0..SIZE {
            b[offset_b + ai * b_stride] = a[offset_a + ai];
        }
    }
}

#[inline(always)]
pub fn interleave_copy_generic<T>(a: &[T], b: &mut [T], size: usize, b_stride: usize)
where
    T: Copy,
{
    for bi in 0..b_stride {
        let offset_a = bi * size;
        let offset_b = bi;
        for ai in 0..size {
            b[offset_b + ai * b_stride] = a[offset_a + ai];
        }
    }
}

#[inline(always)]
pub fn interleave_copy_split<T>(
    real_a: &[T],
    imag_a: &[T],
    real_b: &mut [T],
    imag_b: &mut [T],
    size: usize,
    b_stride: usize,
) where
    T: Copy,
{
    for bi in 0..b_stride {
        let offset_a = bi * size;
        let offset_b = bi;
        for ai in 0..size {
            real_b[offset_b + ai * b_stride] = real_a[offset_a + ai];
            imag_b[offset_b + ai * b_stride] = imag_a[offset_a + ai];
        }
    }
}

#[inline(always)]
pub fn mul<T>(a: Complex<T>, b: Complex<T>, conjugate_second: bool) -> Complex<T>
where
    T: Sample
        + core::ops::Mul<Output = T>
        + core::ops::Add<Output = T>
        + core::ops::Sub<Output = T>
        + Copy
        + std::fmt::Debug,
{
    #[cfg(debug_assertions)]
    {
        if a.re.is_nan() || a.im.is_nan() || b.re.is_nan() || b.im.is_nan() {
            panic!(
                "NaN detected in complex multiplication: a = {:?}, b = {:?}",
                a, b
            );
        }
    }

    if conjugate_second {
        Complex::new(a.re * b.re + a.im * b.im, a.im * b.re - a.re * b.im)
    } else {
        Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
    }
}

#[inline(always)]
pub fn subtract_non_negative<T>(a: T, b: T) -> T
where
    T: core::ops::Sub<Output = T> + PartialOrd + Copy + num::Zero + num::Saturating,
{
    if a > b { a - b } else { T::zero() }
}

#[inline(always)]
pub fn bool_to_value(b: bool) -> usize {
    if b { 1 } else { 0 }
}

#[inline]
pub fn post_decrement<T>(var: &mut T) -> T
where
    T: core::ops::Sub<Output = T> + PartialOrd + Copy + num::Zero + num::One,
{
    let current_value = *var;
    if *var > T::zero() {
        *var = *var - T::one();
    } else {
        *var = T::zero();
    }
    current_value
}

#[inline]
pub fn pre_increment<T>(var: &mut T, max: T) -> T
where
    T: core::ops::Add<Output = T> + PartialOrd + Copy + num::Zero + num::One,
{
    if *var < max {
        *var = *var + T::one();
    }
    *var
}

#[inline]
pub fn post_increment<T>(var: &mut T, max: T) -> T
where
    T: core::ops::Add<Output = T> + PartialOrd + Copy + num::Zero + num::One,
{
    let current_value = *var;
    if *var < max {
        *var = *var + T::one();
    }
    current_value
}

#[inline]
pub fn max_mag<T>(a: Complex<T>, b: Complex<T>) -> Complex<T>
where
    T: Sample + core::ops::Mul<Output = T> + core::ops::Add<Output = T> + Copy,
{
    if a.norm_sqr() >= b.norm_sqr() { a } else { b }
}