use alloc::{vec, vec::Vec};
use num::complex::Complex;

use super::Pow2FFT;
use crate::{dsp::ComplexTrait as _, misc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepType {
    Passthrough,
    InterleaveOrder2,
    InterleaveOrder3,
    InterleaveOrder4,
    InterleaveOrder5,
    InterleaveOrderN,
    FirstFFT,
    MiddleFFT,
    Twiddles,
    FinalOrder2,
    FinalOrder3,
    FinalOrder4,
    FinalOrder5,
    FinalOrderN,
}

#[derive(Debug, Clone, Copy)]
pub struct Step {
    pub stype: StepType,
    pub offset: usize,
}

impl Default for Step {
    fn default() -> Self {
        Step {
            stype: StepType::Passthrough,
            offset: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SplitFFT<T: super::Sample> {
    pub inner_fft: Pow2FFT<T>,

    pub inner_size: usize,
    pub outer_size: usize,

    pub temp_freq: Vec<Complex<T>>,
    pub outer_twiddles: Vec<Complex<T>>,
    pub dft_twists: Vec<Complex<T>>,
    pub dft_tmp: Vec<Complex<T>>,

    pub plan: Vec<Step>,
    pub split_computation: bool,
}

impl<T: super::Sample> Default for SplitFFT<T> {
    fn default() -> Self {
        Self {
            inner_fft: Pow2FFT::default(),

            inner_size: 0,
            outer_size: 0,

            temp_freq: vec![Complex::default(); 0],
            outer_twiddles: vec![Complex::default(); 0],
            dft_twists: vec![Complex::default(); 0],
            dft_tmp: vec![Complex::default(); 0],

            plan: Vec::new(),
            split_computation: true,
        }
    }
}

pub(crate) fn fast_size_above(size: usize) -> usize {
    let mut pow2 = 1;
    while pow2 < 16 && pow2 < size {
        pow2 *= 2;
    }

    while pow2 * 8 < size {
        pow2 *= 2;
    }

    let mut multiple = (size + pow2 - 1) / pow2;
    if multiple == 7 {
        multiple += 1;
    }

    multiple * pow2
}

impl<T: super::Sample> SplitFFT<T> {
    pub fn new(size: usize) -> Self {
        let mut fft = SplitFFT::default();
        fft.resize(size);

        fft
    }

    pub fn get_max_split(&self) -> usize {
        if self.split_computation { 4 } else { 1 }
    }

    pub fn fast_size_above(size: usize) -> usize {
        fast_size_above(size)
    }

    pub const fn max_split() -> usize {
        1
    }

    pub const fn max_inner_size() -> usize {
        16
    }

    pub fn resize(&mut self, size: usize) {
        self.inner_size = 1;
        self.outer_size = size;

        self.dft_tmp.resize(0, Complex::default());
        self.dft_twists.resize(0, Complex::default());
        self.plan.resize(0, Step::default());

        if size == 0 {
            return;
        }

        // Inner size = largest of 2 such that either the inner size >= min inner size
        while (self.outer_size & 1) == 0
            && (self.outer_size > Self::max_split() || self.inner_size < Self::max_inner_size())
        {
            self.inner_size *= 2;
            self.outer_size /= 2;
        }

        self.temp_freq.resize(size, Complex::default());
        self.inner_fft.resize(self.inner_size);

        self.outer_twiddles
            .resize(self.inner_size * (self.outer_size - 1), Complex::default());

        for i in 0..self.inner_size {
            for s in 1..self.outer_size {
                let twiddle_phase = T::from_f32(
                    -2.0 * core::f32::consts::PI * (i as f32) / (self.inner_size as f32)
                        * (s as f32)
                        / (self.outer_size as f32),
                )
                .unwrap();

                self.outer_twiddles[i + (s - 1) * self.inner_size] =
                    Complex::from_polar(T::one(), twiddle_phase);
            }
        }

        let (interleave_step, final_step) = match self.outer_size {
            2 => (StepType::InterleaveOrder2, StepType::FinalOrder2),
            3 => (StepType::InterleaveOrder3, StepType::FinalOrder3),
            4 => (StepType::InterleaveOrder4, StepType::FinalOrder4),
            5 => (StepType::InterleaveOrder5, StepType::FinalOrder5),
            _ => (StepType::InterleaveOrderN, StepType::FinalOrderN),
        };

        if self.outer_size <= 1 {
            if size > 0 {
                self.plan.push(Step {
                    stype: StepType::Passthrough,
                    offset: 0,
                });
            }
        } else {
            self.plan.push(Step {
                stype: interleave_step,
                offset: 0,
            });

            self.plan.push(Step {
                stype: StepType::FirstFFT,
                offset: 0,
            });

            for s in 1..self.outer_size {
                self.plan.push(Step {
                    stype: StepType::MiddleFFT,
                    offset: s * self.inner_size,
                });
            }

            self.plan.push(Step {
                stype: StepType::Twiddles,
                offset: 0,
            });

            self.plan.push(Step {
                stype: final_step,
                offset: 0,
            });

            if final_step == StepType::FinalOrderN {
                self.dft_tmp.resize(self.outer_size, Complex::default());
                self.dft_twists.resize(self.outer_size, Complex::default());

                for s in 0..self.outer_size {
                    let dft_phase = T::from_f32(
                        -2.0 * core::f32::consts::PI * (s as f32) / (self.outer_size as f32),
                    )
                    .unwrap();

                    self.dft_twists[s] = Complex::from_polar(T::one(), dft_phase);
                }
            }
        }
    }

    pub fn size(&self) -> usize {
        self.inner_size * self.outer_size
    }

    pub fn steps(&self) -> usize {
        self.plan.len()
    }

    pub fn fft(&mut self, time: &[Complex<T>], freq: &mut [Complex<T>]) {
        for i in 0..self.plan.len() {
            self.fft_step::<false>(self.plan[i], time, freq);
        }
    }

    pub fn fft_ex(&mut self, step: usize, time: &[Complex<T>], freq: &mut [Complex<T>]) {
        self.fft_step::<false>(self.plan[step], time, freq);
    }

    pub fn fft_split(&mut self, in_r: &[T], in_i: &[T], out_r: &mut [T], out_i: &mut [T]) {
        for i in 0..self.plan.len() {
            self.fft_step_split::<false>(self.plan[i], in_r, in_i, out_r, out_i);
        }
    }

    pub fn fft_split_ex(
        &mut self,
        step: usize,
        in_r: &[T],
        in_i: &[T],
        out_r: &mut [T],
        out_i: &mut [T],
    ) {
        self.fft_step_split::<false>(self.plan[step], in_r, in_i, out_r, out_i);
    }

    pub fn ifft(&mut self, freq: &[Complex<T>], time: &mut [Complex<T>]) {
        for step in (0..self.plan.len()).rev() {
            self.fft_step::<true>(self.plan[step], freq, time);
        }
    }

    pub fn ifft_ex(&mut self, step: usize, freq: &[Complex<T>], time: &mut [Complex<T>]) {
        self.fft_step::<true>(self.plan[step], freq, time);
    }

    pub fn ifft_split(&mut self, in_r: &[T], in_i: &[T], out_r: &mut [T], out_i: &mut [T]) {
        for i in (0..self.plan.len()).rev() {
            self.fft_step_split::<true>(self.plan[i], in_r, in_i, out_r, out_i);
        }
    }

    pub fn ifft_split_ex(
        &mut self,
        step: usize,
        in_r: &[T],
        in_i: &[T],
        out_r: &mut [T],
        out_i: &mut [T],
    ) {
        self.fft_step_split::<true>(self.plan[step], in_r, in_i, out_r, out_i);
    }
}

impl<T: super::Sample> SplitFFT<T> {
    pub(crate) fn fft_step<const INVERSE: bool>(
        &mut self,
        step: Step,
        time: &[Complex<T>],
        freq: &mut [Complex<T>],
    ) {
        match step.stype {
            StepType::Passthrough => {
                if INVERSE {
                    self.inner_fft.ifft(time, freq);
                } else {
                    self.inner_fft.fft(time, freq);
                }
            }

            StepType::InterleaveOrder2 => {
                misc::interleave_copy::<2, Complex<T>>(time, &mut self.temp_freq, self.inner_size);
            }

            StepType::InterleaveOrder3 => {
                misc::interleave_copy::<3, Complex<T>>(time, &mut self.temp_freq, self.inner_size);
            }

            StepType::InterleaveOrder4 => {
                misc::interleave_copy::<4, Complex<T>>(time, &mut self.temp_freq, self.inner_size);
            }

            StepType::InterleaveOrder5 => {
                misc::interleave_copy::<5, Complex<T>>(time, &mut self.temp_freq, self.inner_size);
            }

            StepType::InterleaveOrderN => {
                misc::interleave_copy_generic(
                    time,
                    &mut self.temp_freq,
                    self.outer_size,
                    self.inner_size,
                );
            }

            StepType::FirstFFT => {
                if INVERSE {
                    self.inner_fft.ifft(&self.temp_freq, freq);
                } else {
                    self.inner_fft.fft(&self.temp_freq, freq);
                }
            }

            StepType::MiddleFFT => {
                let offset_out = &mut freq[step.offset..step.offset + self.inner_size];
                let temp_freq = &self.temp_freq[step.offset..step.offset + self.inner_size];

                if INVERSE {
                    self.inner_fft.ifft(temp_freq, offset_out);
                } else {
                    self.inner_fft.fft(temp_freq, offset_out);
                }
            }

            StepType::Twiddles => {
                let freq = &mut freq[self.inner_size..];
                let outer_twiddles = &self.outer_twiddles;
                let size = self.inner_size * (self.outer_size - 1);

                if INVERSE {
                    misc::complex_mul_conj_single(freq, outer_twiddles, size);
                } else {
                    misc::complex_mul_single(freq, outer_twiddles, size);
                }
            }

            StepType::FinalOrder2 => {
                self.final_pass2(freq);
            }

            StepType::FinalOrder3 => {
                self.final_pass3::<INVERSE>(freq);
            }

            StepType::FinalOrder4 => {
                self.final_pass4::<INVERSE>(freq);
            }

            StepType::FinalOrder5 => {
                self.final_pass_5::<INVERSE>(freq);
            }

            StepType::FinalOrderN => {
                self.final_pass_n::<INVERSE>(freq);
            }
        }
    }

    pub(crate) fn fft_step_split<const INVERSE: bool>(
        &mut self,
        step: Step,
        time_r: &[T],
        time_i: &[T],
        freq_r: &mut [T],
        freq_i: &mut [T],
    ) {
        let (temp_freq_r, temp_freq_i) = misc::make_split_complex_mut(&mut self.temp_freq);

        match step.stype {
            StepType::Passthrough => {
                if INVERSE {
                    self.inner_fft.ifft_split(time_r, time_i, freq_r, freq_i);
                } else {
                    self.inner_fft.fft_split(time_r, time_i, freq_r, freq_i);
                }
            }

            StepType::InterleaveOrder2 => {
                misc::interleave_copy::<2, T>(time_r, temp_freq_r, self.inner_size);
                misc::interleave_copy::<2, T>(time_i, temp_freq_i, self.inner_size);
            }

            StepType::InterleaveOrder3 => {
                misc::interleave_copy::<3, T>(time_r, temp_freq_r, self.inner_size);
                misc::interleave_copy::<3, T>(time_i, temp_freq_i, self.inner_size);
            }

            StepType::InterleaveOrder4 => {
                misc::interleave_copy::<4, T>(time_r, temp_freq_r, self.inner_size);
                misc::interleave_copy::<4, T>(time_i, temp_freq_i, self.inner_size);
            }

            StepType::InterleaveOrder5 => {
                misc::interleave_copy::<5, T>(time_r, temp_freq_r, self.inner_size);
                misc::interleave_copy::<5, T>(time_i, temp_freq_i, self.inner_size);
            }

            StepType::InterleaveOrderN => {
                misc::interleave_copy_split(
                    time_r,
                    time_i,
                    freq_r,
                    freq_i,
                    self.outer_size,
                    self.inner_size,
                );
            }

            StepType::FirstFFT => {
                if INVERSE {
                    self.inner_fft
                        .ifft_split(temp_freq_r, temp_freq_i, freq_r, freq_i);
                } else {
                    self.inner_fft
                        .fft_split(temp_freq_r, temp_freq_i, freq_r, freq_i);
                }
            }

            StepType::MiddleFFT => {
                let offset_out_r = &mut freq_r[step.offset..step.offset + self.inner_size];
                let offset_out_i = &mut freq_i[step.offset..step.offset + self.inner_size];
                let temp_freq_r = &temp_freq_r[step.offset..step.offset + self.inner_size];
                let temp_freq_i = &temp_freq_i[step.offset..step.offset + self.inner_size];

                if INVERSE {
                    self.inner_fft
                        .ifft_split(temp_freq_r, temp_freq_i, offset_out_r, offset_out_i);
                } else {
                    self.inner_fft
                        .fft_split(temp_freq_r, temp_freq_i, offset_out_r, offset_out_i);
                }
            }

            StepType::Twiddles => {
                let freq_r = &mut freq_r[self.inner_size..];
                let freq_i = &mut freq_i[self.inner_size..];
                let (twiddles_r, twiddles_i) = misc::make_split_complex(&self.outer_twiddles);

                if INVERSE {
                    misc::complex_mul_conj_split_single(
                        freq_r,
                        freq_i,
                        twiddles_r,
                        twiddles_i,
                        self.inner_size * (self.outer_size - 1),
                    );
                } else {
                    misc::complex_mul_split_single(
                        freq_r,
                        freq_i,
                        twiddles_r,
                        twiddles_i,
                        self.inner_size * (self.outer_size - 1),
                    );
                }
            }

            StepType::FinalOrder2 => {
                self.final_pass2_split(freq_r, freq_i);
            }

            StepType::FinalOrder3 => {
                self.final_pass3_split::<INVERSE>(freq_r, freq_i);
            }

            StepType::FinalOrder4 => {
                self.final_pass4_split::<INVERSE>(freq_r, freq_i);
            }

            StepType::FinalOrder5 => {
                self.final_pass_5_split::<INVERSE>(freq_r, freq_i);
            }

            StepType::FinalOrderN => {
                self.final_pass_n_split::<INVERSE>(freq_r, freq_i);
            }
        }
    }

    pub(crate) fn final_pass2(&mut self, f0: &mut [Complex<T>]) {
        let (f0, f1) = f0.split_at_mut(self.inner_size);
        for i in 0..self.inner_size {
            let a = f0[i];
            let b = f1[i];

            f0[i] = a + b;
            f1[i] = a - b;
        }
    }

    pub(crate) fn final_pass2_split(&mut self, f0_r: &mut [T], f0_i: &mut [T]) {
        let (f0r, f1r) = f0_r.split_at_mut(self.inner_size);
        let (f0i, f1i) = f0_i.split_at_mut(self.inner_size);

        for i in 0..self.inner_size {
            let ar = f0r[i];
            let ai = f0i[i];
            let br = f1r[i];
            let bi = f1i[i];

            f0r[i] = ar + br;
            f0i[i] = ai + bi;
            f1r[i] = ar - br;
            f1i[i] = ai - bi;
        }
    }

    pub(crate) fn final_pass3<const INVERSE: bool>(&mut self, f0: &mut [Complex<T>]) {
        let tw1 = Complex {
            re: T::from_f32(-0.5).unwrap(),
            im: if INVERSE {
                T::from_f32(TW1_INV.im).unwrap()
            } else {
                T::from_f32(TW1_FWD.im).unwrap()
            },
        };

        let (f0, f1) = f0.split_at_mut(self.inner_size);
        let (f1, f2) = f1.split_at_mut(self.inner_size);

        for i in 0..self.inner_size {
            let a = f0[i];
            let b = f1[i];
            let c = f2[i];

            let bc0 = b + c;
            let bc1 = b - c;

            f0[i] = a + bc0;
            f1[i] = Complex {
                re: a.real() + bc0.real() * tw1.real() - bc1.imag() * tw1.imag(),
                im: a.imag() + bc0.imag() * tw1.real() + bc1.real() * tw1.imag(),
            };
            f2[i] = Complex {
                re: a.real() + bc0.real() * tw1.real() + bc1.imag() * tw1.imag(),
                im: a.imag() + bc0.imag() * tw1.real() - bc1.real() * tw1.imag(),
            };
        }
    }

    pub(crate) fn final_pass3_split<const INVERSE: bool>(
        &mut self,
        f0_r: &mut [T],
        f0_i: &mut [T],
    ) {
        let tw1r = T::from_f32(-0.5).unwrap();
        let tw1i = if INVERSE {
            T::from_f32(TW1_INV.im).unwrap()
        } else {
            T::from_f32(TW1_FWD.im).unwrap()
        };

        let (f0r, f1r) = f0_r.split_at_mut(self.inner_size);
        let (f1r, f2r) = f1r.split_at_mut(self.inner_size);
        let (f0i, f1i) = f0_i.split_at_mut(self.inner_size);
        let (f1i, f2i) = f1i.split_at_mut(self.inner_size);

        for i in 0..self.inner_size {
            let ar = f0r[i];
            let ai = f0i[i];
            let br = f1r[i];
            let bi = f1i[i];
            let cr = f2r[i];
            let ci = f2i[i];

            f0r[i] = ar + br + cr;
            f0i[i] = ai + bi + ci;
            f1r[i] = ar + br * tw1r - bi * tw1i + cr * tw1r + ci * tw1i;
            f1i[i] = ai + bi * tw1r + br * tw1i - cr * tw1i + ci * tw1r;
            f2r[i] = ar + br * tw1r + bi * tw1i + cr * tw1r - ci * tw1i;
            f2i[i] = ai + bi * tw1r - br * tw1i + cr * tw1i + ci * tw1r;
        }
    }

    pub(crate) fn final_pass4<const INVERSE: bool>(&mut self, f0: &mut [Complex<T>]) {
        let (f0, f1) = f0.split_at_mut(self.inner_size);
        let (f1, f2) = f1.split_at_mut(self.inner_size);
        let (f2, f3) = f2.split_at_mut(self.inner_size);

        for i in 0..self.inner_size {
            let a = f0[i];
            let b = f1[i];
            let c = f2[i];
            let d = f3[i];

            let ac0 = a + c;
            let ac1 = a - c;
            let bd0 = b + d;
            let bd1 = if INVERSE { b - d } else { d - b };

            let bd1i = Complex {
                re: -bd1.im,
                im: bd1.re,
            };

            f0[i] = ac0 + bd0;
            f1[i] = ac1 + bd1i;
            f2[i] = ac0 - bd0;
            f3[i] = ac1 - bd1i;
        }
    }

    pub(crate) fn final_pass4_split<const INVERSE: bool>(
        &mut self,
        f0_r: &mut [T],
        f0_i: &mut [T],
    ) {
        let (f0r, f1r) = f0_r.split_at_mut(self.inner_size);
        let (f1r, f2r) = f1r.split_at_mut(self.inner_size);
        let (f2r, f3r) = f2r.split_at_mut(self.inner_size);
        let (f0i, f1i) = f0_i.split_at_mut(self.inner_size);
        let (f1i, f2i) = f1i.split_at_mut(self.inner_size);
        let (f2i, f3i) = f2i.split_at_mut(self.inner_size);

        for i in 0..self.inner_size {
            let ar = f0r[i];
            let ai = f0i[i];
            let br = f1r[i];
            let bi = f1i[i];
            let cr = f2r[i];
            let ci = f2i[i];
            let dr = f3r[i];
            let di = f3i[i];

            let ac0r = ar + cr;
            let ac0i = ai + ci;
            let ac1r = ar - cr;
            let ac1i = ai - ci;
            let bd0r = br + dr;
            let bd0i = bi + di;
            let bd1r = br - dr;
            let bd1i = bi - di;

            f0r[i] = ac0r + bd0r;
            f0i[i] = ac0i + bd0i;
            f1r[i] = if INVERSE { ac1r - bd1i } else { ac1r + bd1i };
            f1i[i] = if INVERSE { ac1i + bd1r } else { ac1i - bd1r };
            f2r[i] = ac0r - bd0r;
            f2i[i] = ac0i - bd0i;
            f3r[i] = if INVERSE { ac1r + bd1i } else { ac1r - bd1i };
            f3i[i] = if INVERSE { ac1i - bd1r } else { ac1i + bd1r };
        }
    }

    pub(crate) fn final_pass_5<const INVERSE: bool>(&mut self, f0: &mut [Complex<T>]) {
        let tw1r = T::from_f32(TW1_R).unwrap();
        let tw1i = if INVERSE {
            T::from_f32(TW2_I_INV).unwrap()
        } else {
            T::from_f32(TW2_I_FWD).unwrap()
        };
        let tw2r = T::from_f32(TW2_R).unwrap();
        let tw2i = if INVERSE {
            T::from_f32(TW3_I_INV).unwrap()
        } else {
            T::from_f32(TW3_I_FWD).unwrap()
        };

        let (f0, f1) = f0.split_at_mut(self.inner_size);
        let (f1, f2) = f1.split_at_mut(self.inner_size);
        let (f2, f3) = f2.split_at_mut(self.inner_size);
        let (f3, f4) = f3.split_at_mut(self.inner_size);

        for i in 0..self.inner_size {
            let a = f0[i];
            let b = f1[i];
            let c = f2[i];
            let d = f3[i];
            let e = f4[i];

            let be0 = b + e;
            let be1 = Complex {
                re: e.im - b.im,
                im: b.re - e.re,
            };
            let cd0 = c + d;
            let cd1 = Complex {
                re: d.im - c.im,
                im: c.re - d.re,
            };

            let bcde01 = be0 * tw1r + cd0 * tw2r;
            let bcde02 = be0 * tw2r + cd0 * tw1r;
            let bcde11 = be1 * tw1i + cd1 * tw2i;
            let bcde12 = be1 * tw2i - cd1 * tw1i;

            f0[i] = a + be0 + cd0;
            f1[i] = a + bcde01 + bcde11;
            f2[i] = a + bcde02 + bcde12;
            f3[i] = a + bcde02 - bcde12;
            f4[i] = a + bcde01 - bcde11;
        }
    }

    pub(crate) fn final_pass_5_split<const INVERSE: bool>(
        &mut self,
        f0_r: &mut [T],
        f0_i: &mut [T],
    ) {
        let tw1r = T::from_f32(TW1_R).unwrap();
        let tw1i = if INVERSE {
            T::from_f32(TW2_I_INV).unwrap()
        } else {
            T::from_f32(TW2_I_FWD).unwrap()
        };
        let tw2r = T::from_f32(TW2_R).unwrap();
        let tw2i = if INVERSE {
            T::from_f32(TW3_I_INV).unwrap()
        } else {
            T::from_f32(TW3_I_FWD).unwrap()
        };

        let (f0r, f1r) = f0_r.split_at_mut(self.inner_size);
        let (f1r, f2r) = f1r.split_at_mut(self.inner_size);
        let (f2r, f3r) = f2r.split_at_mut(self.inner_size);
        let (f3r, f4r) = f3r.split_at_mut(self.inner_size);
        let (f0i, f1i) = f0_i.split_at_mut(self.inner_size);
        let (f1i, f2i) = f1i.split_at_mut(self.inner_size);
        let (f2i, f3i) = f2i.split_at_mut(self.inner_size);
        let (f3i, f4i) = f3i.split_at_mut(self.inner_size);

        for i in 0..self.inner_size {
            let ar = f0r[i];
            let ai = f0i[i];
            let br = f1r[i];
            let bi = f1i[i];
            let cr = f2r[i];
            let ci = f2i[i];
            let dr = f3r[i];
            let di = f3i[i];
            let er = f4r[i];
            let ei = f4i[i];

            let be0r = br + er;
            let be0i = bi + ei;
            let be1r = ei - bi;
            let be1i = br - er;
            let cd0r = cr + dr;
            let cd0i = ci + di;
            let cd1r = di - ci;
            let cd1i = cr - dr;

            let bcde01r = be0r * tw1r + cd0r * tw2r;
            let bcde01i = be0i * tw1r + cd0i * tw2r;
            let bcde02r = be0r * tw2r + cd0r * tw1r;
            let bcde02i = be0i * tw2r + cd0i * tw1r;
            let bcde11r = be1r * tw1i + cd1r * tw2i;
            let bcde11i = be1i * tw1i + cd1i * tw2i;
            let bcde12r = be1r * tw2i - cd1r * tw1i;
            let bcde12i = be1i * tw2i - cd1i * tw1i;

            f0r[i] = ar + be0r + cd0r;
            f0i[i] = ai + be0i + cd0i;
            f1r[i] = ar + bcde01r + bcde11r;
            f1i[i] = ai + bcde01i + bcde11i;
            f2r[i] = ar + bcde02r + bcde12r;
            f2i[i] = ai + bcde02i + bcde12i;
            f3r[i] = ar + bcde02r - bcde12r;
            f3i[i] = ai + bcde02i - bcde12i;
            f4r[i] = ar + bcde01r - bcde11r;
            f4i[i] = ai + bcde01i - bcde11i;
        }
    }

    pub(crate) fn final_pass_n<const INVERSE: bool>(&mut self, f0: &mut [Complex<T>]) {
        for i in 0..self.inner_size {
            let offset_freq = &mut f0[i..];

            let mut sum = Complex::<T>::default();
            for i2 in 0..self.outer_size {
                let value = offset_freq[i2 * self.inner_size];
                self.dft_tmp[i2] = value;
                sum = sum + value;
            }

            offset_freq[0] = sum;

            for f in 1..self.outer_size {
                let mut sum = self.dft_tmp[0];

                for i2 in 1..self.outer_size {
                    let twist_index = (f * i2) % self.outer_size;
                    let twist = if INVERSE {
                        self.dft_twists[twist_index].conj()
                    } else {
                        self.dft_twists[twist_index]
                    };

                    sum = sum
                        + Complex {
                            re: self.dft_tmp[i2].re * twist.re - self.dft_tmp[i2].im * twist.im,
                            im: self.dft_tmp[i2].re * twist.im + self.dft_tmp[i2].im * twist.re,
                        };
                }

                offset_freq[f * self.inner_size] = sum;
            }
        }
    }

    pub(crate) fn final_pass_n_split<const INVERSE: bool>(
        &mut self,
        f0_r: &mut [T],
        f0_i: &mut [T],
    ) {
        let (dft_tmp_r, dft_tmp_i) = misc::make_split_complex_mut(&mut self.dft_tmp);

        for i in 0..self.inner_size {
            let offset_r = &mut f0_r[i..];
            let offset_i = &mut f0_i[i..];

            let mut sum_r = T::zero();
            let mut sum_i = T::zero();

            for i2 in 0..self.outer_size {
                let val_r = offset_r[i2 * self.inner_size];
                let val_i = offset_i[i2 * self.inner_size];

                dft_tmp_r[i2] = val_r;
                dft_tmp_i[i2] = val_i;

                sum_r = sum_r + val_r;
                sum_i = sum_i + val_i;
            }

            offset_r[0] = sum_r;
            offset_i[0] = sum_i;

            for f in 1..self.outer_size {
                for i2 in 0..self.outer_size {
                    let twist_index = (f * i2) % self.outer_size;
                    let twist = if INVERSE {
                        self.dft_twists[twist_index].conj()
                    } else {
                        self.dft_twists[twist_index]
                    };

                    sum_r = sum_r + (dft_tmp_r[i2] * twist.re - dft_tmp_i[i2] * twist.im);
                    sum_i = sum_i + (dft_tmp_r[i2] * twist.im + dft_tmp_i[i2] * twist.re);
                }

                offset_r[f * self.inner_size] = sum_r;
                offset_i[f * self.inner_size] = sum_i;
            }
        }
    }
}

const TW1_INV: Complex<f32> = Complex {
    re: -0.5,
    im: -0.8660254037844386 * -1.0, // -sqrt(0.75) * -1.0
};

const TW1_FWD: Complex<f32> = Complex {
    re: -0.5,
    im: -0.8660254037844386 * 1.0, // -sqrt(0.75) * 1.0
};

const TW1_R: f32 = 0.30901699437494745;
const TW2_I_INV: f32 = -0.9510565162951535 * -1.0;
const TW2_I_FWD: f32 = -0.9510565162951535 * 1.0;
const TW2_R: f32 = -0.8090169943749473;
const TW3_I_INV: f32 = -0.5877852522924732 * -1.0;
const TW3_I_FWD: f32 = -0.5877852522924732 * 1.0;
