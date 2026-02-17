use alloc::{vec, vec::Vec};
use num::complex::Complex;

use super::SplitFFT;
use crate::misc;

#[derive(Debug, Clone)]
pub struct RealFFT<
    T: super::Sample,
    const SPLIT_COMPUTATION: bool = false,
    const PREFER_COMPLEX_SPLIT: bool = false,
> {
    pub complex_fft: SplitFFT<T>,

    pub temp_freq: Vec<Complex<T>>,
    pub temp_time: Vec<Complex<T>>,

    pub twiddles: Vec<Complex<T>>,
    pub half_bin_twists: Vec<Complex<T>>,
    pub half_bin_shift: bool,
}

impl<T: super::Sample, const SPLIT_COMPUTATION: bool, const PREFER_COMPLEX_SPLIT: bool> Default
    for RealFFT<T, SPLIT_COMPUTATION, PREFER_COMPLEX_SPLIT>
{
    fn default() -> Self {
        Self {
            complex_fft: SplitFFT::new(0),
            temp_freq: vec![],
            temp_time: vec![],
            twiddles: vec![],
            half_bin_twists: vec![],
            half_bin_shift: false,
        }
    }
}

impl<T: super::Sample, const SPLIT_COMPUTATION: bool, const PREFER_COMPLEX_SPLIT: bool>
    RealFFT<T, SPLIT_COMPUTATION, PREFER_COMPLEX_SPLIT>
{
    pub fn new(size: usize) -> Self {
        let mut fft = RealFFT::default();
        fft.resize(size);

        fft
    }

    pub fn fast_size_above(size: usize) -> usize {
        SplitFFT::<T>::fast_size_above((size + 1) / 2) * 2
    }

    pub fn resize(&mut self, size: usize) {
        let h_size = size / 2;
        self.complex_fft.resize(h_size);
        self.temp_time.resize(h_size, Complex::default());
        self.temp_freq.resize(h_size, Complex::default());
        self.twiddles.resize((h_size / 2) + 1, Complex::default());

        if !self.half_bin_shift {
            for i in 0..self.twiddles.len() {
                let rot_phase = T::from_f32(
                    (i as f32) * (-2.0 * core::f32::consts::PI / (size as f32))
                        - (core::f32::consts::PI / 2.0),
                )
                .unwrap();

                self.twiddles[i] = Complex::from_polar(T::one(), rot_phase);
            }
        } else {
            for i in 0..self.twiddles.len() {
                let rot_phase = T::from_f32(
                    ((i as f32) + 0.5) * (-2.0 * core::f32::consts::PI / (size as f32))
                        - (core::f32::consts::PI / 2.0),
                )
                .unwrap();

                self.twiddles[i] = Complex::from_polar(T::one(), rot_phase);
            }

            self.half_bin_twists.resize(h_size, Complex::default());

            for i in 0..h_size {
                let twist_phase =
                    T::from_f32((-2.0 * core::f32::consts::PI * (i as f32)) / (size as f32))
                        .unwrap();

                self.half_bin_twists[i] = Complex::from_polar(T::one(), twist_phase);
            }
        }
    }

    pub fn set_half_bin_shift(&mut self, half_bin_shift: bool) {
        if self.half_bin_shift != half_bin_shift {
            self.half_bin_shift = half_bin_shift;
            self.resize(self.size());
        }
    }

    pub fn size(&self) -> usize {
        self.complex_fft.size() * 2
    }

    pub fn steps(&self) -> usize {
        self.complex_fft.steps() + if SPLIT_COMPUTATION { 3 } else { 2 }
    }

    pub fn fft(&mut self, time: &[T], freq: &mut [Complex<T>]) {
        for step in 0..self.steps() {
            self.fft_ex(step, time, freq);
        }
    }

    pub fn fft_ex(&mut self, mut step: usize, time: &[T], freq: &mut [Complex<T>]) {
        if PREFER_COMPLEX_SPLIT {
            let h_size = self.complex_fft.size();
            let (temp_time_r, temp_time_i) = misc::make_split_complex_mut(&mut self.temp_time);
            let (temp_freq_r, temp_freq_i) = misc::make_split_complex_mut(&mut self.temp_freq);

            if misc::post_decrement(&mut step) == 0 {
                if self.half_bin_shift {
                    for i in 0..h_size {
                        let tr = time[2 * i];
                        let ti = time[2 * i + 1];
                        let twist = self.half_bin_twists[i];
                        temp_time_r[i] = tr * twist.re - ti * twist.im;
                        temp_time_i[i] = ti * twist.re + tr * twist.im;
                    }
                } else {
                    for i in 0..h_size {
                        temp_time_r[i] = time[2 * i];
                        temp_time_i[i] = time[2 * i + 1];
                    }
                }
            } else if step < self.complex_fft.steps() {
                self.complex_fft.fft_split_ex(
                    step,
                    temp_time_r,
                    temp_time_i,
                    temp_freq_r,
                    temp_freq_i,
                );
            } else {
                if !self.half_bin_shift {
                    let bin0r = temp_freq_r[0];
                    let bin0i = temp_freq_i[0];
                    freq[0] = Complex {
                        re: bin0r + bin0i,
                        im: bin0r - bin0i,
                    };
                }

                let mut start_i = if self.half_bin_shift { 0 } else { 1 };
                let mut end_i = h_size / 2 + 1;

                if SPLIT_COMPUTATION {
                    if step == self.complex_fft.steps() {
                        end_i = (start_i + end_i) / 2;
                    } else {
                        start_i = (start_i + end_i) / 2;
                    }
                }

                let half_val = T::from_f32(0.5).unwrap();

                for i in start_i..end_i {
                    let conj_i = if self.half_bin_shift {
                        h_size - 1 - i
                    } else {
                        h_size - i
                    };

                    let twiddle = self.twiddles[i];

                    let odd_r = (temp_freq_r[i] + temp_freq_r[conj_i]) * half_val;
                    let odd_i = (temp_freq_i[i] - temp_freq_i[conj_i]) * half_val;
                    let eveni_r = (temp_freq_r[i] - temp_freq_r[conj_i]) * half_val;
                    let eveni_i = (temp_freq_i[i] + temp_freq_i[conj_i]) * half_val;
                    let even_rot_minus_ir = eveni_r * twiddle.re - eveni_i * twiddle.im;
                    let even_rot_minus_ii = eveni_r * twiddle.re + eveni_i * twiddle.im;

                    freq[i] = Complex {
                        re: odd_r + even_rot_minus_ir,
                        im: odd_i + even_rot_minus_ii,
                    };
                    freq[conj_i] = Complex {
                        re: odd_r - even_rot_minus_ir,
                        im: even_rot_minus_ii - odd_i,
                    };
                }
            }
        } else {
            // let can_use_time = !self.half_bin_shift && misc::is_align_array(time, core::mem::align_of::<Complex<T>>());
            let h_size = self.complex_fft.size();

            if misc::post_decrement(&mut step) == 0 {
                if self.half_bin_shift {
                    for i in 0..h_size {
                        let tr = time[2 * i];
                        let ti = time[2 * i + 1];
                        let twist = self.half_bin_twists[i];
                        self.temp_time[i] = Complex {
                            re: tr * twist.re - ti * twist.im,
                            im: ti * twist.im + tr * twist.re,
                        };
                    }
                } else {
                    for i in 0..h_size {
                        self.temp_time[i] = Complex {
                            re: time[2 * i],
                            im: time[2 * i + 1],
                        };
                    }
                }
            } else if step < self.complex_fft.steps() {
                self.complex_fft
                    .fft_ex(step, &self.temp_time, &mut self.temp_freq);
            } else {
                if !self.half_bin_shift {
                    let bin0 = self.temp_freq[0];
                    freq[0] = Complex {
                        re: bin0.re + bin0.im,
                        im: bin0.re - bin0.im,
                    }
                }

                let h_size = self.complex_fft.size();
                let mut start_i = if self.half_bin_shift { 0 } else { 1 };
                let mut end_i = h_size / 2 + 1;

                if SPLIT_COMPUTATION {
                    if step == self.complex_fft.steps() {
                        end_i = (start_i + end_i) / 2;
                    } else {
                        start_i = (start_i + end_i) / 2;
                    }
                }

                for i in start_i..end_i {
                    let conj_i = if self.half_bin_shift {
                        h_size - 1 - i
                    } else {
                        h_size - i
                    };

                    let twiddle = self.twiddles[i];

                    let odd = (self.temp_freq[i] + Complex::conj(&self.temp_freq[conj_i]))
                        * T::from_f32(0.5).unwrap();
                    let even_i = (self.temp_freq[i] - Complex::conj(&self.temp_freq[conj_i]))
                        * T::from_f32(0.5).unwrap();
                    let even_rot_minus_i = Complex {
                        re: even_i.re * twiddle.re - even_i.im * twiddle.im,
                        im: even_i.im * twiddle.re + even_i.re * twiddle.im,
                    };

                    freq[i] = odd + even_rot_minus_i;
                    freq[conj_i] = Complex {
                        re: odd.re - even_rot_minus_i.re,
                        im: even_rot_minus_i.im - odd.im,
                    };
                }
            }
        }
    }

    pub fn fft_split(&mut self, time_r: &[T], freq_r: &mut [T], freq_i: &mut [T]) {
        for step in 0..self.steps() {
            self.fft_split_ex(step, time_r, freq_r, freq_i);
        }
    }

    pub fn fft_split_ex(
        &mut self,
        mut step: usize,
        time_r: &[T],
        freq_r: &mut [T],
        freq_i: &mut [T],
    ) {
        let h_size = self.complex_fft.size();
        let (temp_time_r, temp_time_i) = misc::make_split_complex_mut(&mut self.temp_time);
        let (temp_freq_r, temp_freq_i) = misc::make_split_complex_mut(&mut self.temp_freq);

        if misc::post_decrement(&mut step) == 0 {
            if self.half_bin_shift {
                for i in 0..h_size {
                    let tr = time_r[2 * i];
                    let ti = time_r[2 * i + 1];
                    let twist = self.half_bin_twists[i];
                    temp_time_r[i] = tr * twist.re - ti * twist.im;
                    temp_time_i[i] = ti * twist.re + tr * twist.im;
                }
            } else {
                for i in 0..h_size {
                    temp_time_r[i] = time_r[2 * i];
                    temp_time_i[i] = time_r[2 * i + 1];
                }
            }
        } else if step < self.complex_fft.steps() {
            self.complex_fft
                .fft_split_ex(step, temp_time_r, temp_time_i, temp_freq_r, temp_freq_i);
        } else {
            if self.half_bin_shift {
                let bin0r = temp_freq_r[0];
                let bin0i = temp_freq_i[0];
                freq_r[0] = bin0r + bin0i;
                freq_i[0] = bin0r - bin0i;
            }

            let mut start_i = if self.half_bin_shift { 0 } else { 1 };
            let mut end_i = h_size / 2 + 1;
            if SPLIT_COMPUTATION {
                if step == self.complex_fft.steps() {
                    end_i = (start_i + end_i) / 2;
                } else {
                    start_i = (start_i + end_i) / 2;
                }
            }

            for i in start_i..end_i {
                let conj_i = if self.half_bin_shift {
                    h_size - 1 - i
                } else {
                    h_size - i
                };

                let twiddle = self.twiddles[i];

                let odd_r = (temp_freq_r[i] + temp_freq_r[conj_i]) * T::from_f32(0.5).unwrap();
                let odd_i = (temp_freq_i[i] - temp_freq_i[conj_i]) * T::from_f32(0.5).unwrap();
                let eveni_r = (temp_freq_r[i] - temp_freq_r[conj_i]) * T::from_f32(0.5).unwrap();
                let eveni_i = (temp_freq_i[i] + temp_freq_i[conj_i]) * T::from_f32(0.5).unwrap();
                let even_rot_minus_ir = eveni_r * twiddle.re - eveni_i * twiddle.im;
                let even_rot_minus_ii = eveni_i * twiddle.re + eveni_r * twiddle.im;

                freq_r[i] = odd_r + even_rot_minus_ir;
                freq_i[i] = odd_i + even_rot_minus_ii;
                freq_r[conj_i] = odd_r - even_rot_minus_ir;
                freq_i[conj_i] = even_rot_minus_ii - odd_i;
            }
        }
    }

    pub fn ifft(&mut self, freq: &[Complex<T>], time: &mut [T]) {
        for step in 0..self.steps() {
            self.ifft_ex(step, freq, time);
        }
    }

    pub fn ifft_ex(&mut self, mut step: usize, freq: &[Complex<T>], time: &mut [T]) {
        if PREFER_COMPLEX_SPLIT {
            let (temp_time_r, temp_time_i) = misc::make_split_complex_mut(&mut self.temp_time);
            let (temp_freq_r, temp_freq_i) = misc::make_split_complex_mut(&mut self.temp_freq);

            let h_size = self.complex_fft.size();
            let split_first = SPLIT_COMPUTATION && misc::post_decrement(&mut step) == 0;

            if split_first || misc::post_decrement(&mut step) == 0 {
                let bin0 = freq[0];
                if !self.half_bin_shift {
                    temp_freq_r[0] = bin0.re + bin0.im;
                    temp_freq_i[0] = bin0.re - bin0.im;
                }

                let mut start_i = if self.half_bin_shift { 0 } else { 1 };
                let mut end_i = h_size / 2 + 1;
                if SPLIT_COMPUTATION {
                    if split_first {
                        end_i = (start_i + end_i) / 2;
                    } else {
                        start_i = (start_i + end_i) / 2;
                    }
                }

                for i in start_i..end_i {
                    let conj_i = if self.half_bin_shift {
                        h_size - 1 - i
                    } else {
                        h_size - i
                    };

                    let twiddle = self.twiddles[i];

                    let odd = freq[i] + Complex::conj(&freq[conj_i]);
                    let even_rot_minus_i = freq[i] - Complex::conj(&freq[conj_i]);
                    let even_i = Complex {
                        re: even_rot_minus_i.re * twiddle.re + even_rot_minus_i.im * twiddle.im,
                        im: even_rot_minus_i.im * twiddle.re - even_rot_minus_i.re * twiddle.im,
                    };

                    temp_freq_r[i] = odd.re + even_i.re;
                    temp_freq_i[i] = odd.im + even_i.im;
                    temp_freq_r[conj_i] = odd.re - even_i.re;
                    temp_freq_i[conj_i] = even_i.im - odd.im;
                }
            } else if step < self.complex_fft.steps() {
                self.complex_fft.ifft_split_ex(
                    step,
                    temp_freq_r,
                    temp_freq_i,
                    temp_time_r,
                    temp_time_i,
                );
            } else {
                if self.half_bin_shift {
                    for i in 0..h_size {
                        let twist = self.half_bin_twists[i];
                        let tr = temp_time_r[i];
                        let ti = temp_time_i[i];
                        time[2 * i] = tr * twist.re + ti * twist.im;
                        time[2 * i + 1] = ti * twist.re - tr * twist.im;
                    }
                } else {
                    for i in 0..h_size {
                        time[2 * i] = temp_time_r[i];
                        time[2 * i + 1] = temp_time_i[i];
                    }
                }
            }
        } else {
            // let can_use_time = !self.half_bin_shift && misc::is_align_array(time, core::mem::align_of::<Complex<T>>());
            let h_size = self.complex_fft.size();
            let split_first_time = SPLIT_COMPUTATION && misc::post_decrement(&mut step) == 0;

            if split_first_time || misc::post_decrement(&mut step) == 0 {
                let bin0 = freq[0];
                if !self.half_bin_shift {
                    self.temp_freq[0] = Complex {
                        re: bin0.re + bin0.im,
                        im: bin0.re - bin0.im,
                    };
                }

                let mut start_i = if self.half_bin_shift { 0 } else { 1 };
                let mut end_i = h_size / 2 + 1;
                if SPLIT_COMPUTATION {
                    if split_first_time {
                        end_i = (start_i + end_i) / 2;
                    } else {
                        start_i = (start_i + end_i) / 2;
                    }
                }

                for i in start_i..end_i {
                    let conj_i = if self.half_bin_shift {
                        h_size - 1 - i
                    } else {
                        h_size - i
                    };

                    let twiddle = self.twiddles[i];

                    let odd = freq[i] + Complex::conj(&freq[conj_i]);
                    let even_rot_minus_i = freq[i] - Complex::conj(&freq[conj_i]);
                    let even_i = Complex {
                        re: even_rot_minus_i.re * twiddle.re + even_rot_minus_i.im * twiddle.im,
                        im: even_rot_minus_i.im * twiddle.re - even_rot_minus_i.re * twiddle.im,
                    };

                    self.temp_freq[i] = odd + even_i;
                    self.temp_freq[conj_i] = Complex {
                        re: odd.re - even_i.re,
                        im: even_i.im - odd.im,
                    };
                }
            } else if step < self.complex_fft.steps() {
                self.complex_fft
                    .ifft_ex(step, &self.temp_freq, &mut self.temp_time);
            } else {
                if self.half_bin_shift {
                    for i in 0..h_size {
                        let twist = self.half_bin_twists[i];
                        let t = self.temp_time[i];

                        time[2 * i] = t.re * twist.re + t.im * twist.im;
                        time[2 * i + 1] = t.im * twist.re - t.re * twist.im;
                    }
                } else {
                    for i in 0..h_size {
                        let t = self.temp_time[i];
                        time[2 * i] = t.re;
                        time[2 * i + 1] = t.im;
                    }
                }
            }
        }
    }

    pub fn ifft_split(&mut self, freq_r: &[T], freq_i: &[T], time_r: &mut [T]) {
        for step in 0..self.steps() {
            self.ifft_split_ex(step, freq_r, freq_i, time_r);
        }
    }

    pub fn ifft_split_ex(&mut self, mut step: usize, freq_r: &[T], freq_i: &[T], time_r: &mut [T]) {
        let h_size = self.complex_fft.size();
        let (temp_time_r, temp_time_i) = misc::make_split_complex_mut(&mut self.temp_time);
        let (temp_freq_r, temp_freq_i) = misc::make_split_complex_mut(&mut self.temp_freq);

        let split_first = SPLIT_COMPUTATION && misc::post_decrement(&mut step) == 0;

        if split_first || misc::post_decrement(&mut step) == 0 {
            let bin0r = freq_r[0];
            let bin0i = freq_i[0];
            if !self.half_bin_shift {
                temp_freq_r[0] = bin0r + bin0i;
                temp_freq_i[0] = bin0r - bin0i;
            }

            let mut start_i = if self.half_bin_shift { 0 } else { 1 };
            let mut end_i = h_size / 2 + 1;

            if SPLIT_COMPUTATION {
                if split_first {
                    end_i = (start_i + end_i) / 2;
                } else {
                    start_i = (start_i + end_i) / 2;
                }
            }

            for i in start_i..end_i {
                let conj_i = if self.half_bin_shift {
                    h_size - 1 - i
                } else {
                    h_size - i
                };

                let twiddle = self.twiddles[i];
                let fir = freq_r[i];
                let fii = freq_i[i];
                let fcir = freq_r[conj_i];
                let fcii = freq_i[conj_i];

                let odd_r = Complex::new(fir + fcir, fii - fcii);
                let even_rot_minus_ir = Complex::new(fir - fcir, fii + fcii);
                let even_i = Complex {
                    re: even_rot_minus_ir.re * twiddle.re + even_rot_minus_ir.im * twiddle.im,
                    im: even_rot_minus_ir.im * twiddle.re - even_rot_minus_ir.re * twiddle.im,
                };

                temp_freq_r[i] = odd_r.re + even_i.re;
                temp_freq_i[i] = odd_r.im + even_i.im;
                temp_freq_r[conj_i] = odd_r.re - even_i.re;
                temp_freq_i[conj_i] = even_i.im - odd_r.im;
            }
        } else if misc::post_decrement(&mut step) < self.complex_fft.steps() {
            self.complex_fft.ifft_split_ex(
                step,
                temp_freq_r,
                temp_freq_i,
                temp_time_r,
                temp_time_i,
            );
        } else {
            if self.half_bin_shift {
                for i in 0..h_size {
                    let tr = temp_time_r[i];
                    let ti = temp_time_i[i];
                    let twist = self.half_bin_twists[i];

                    time_r[2 * i] = tr * twist.re + ti * twist.im;
                    time_r[2 * i + 1] = ti * twist.re - tr * twist.im;
                }
            } else {
                for i in 0..h_size {
                    time_r[2 * i] = temp_time_r[i];
                    time_r[2 * i + 1] = temp_time_i[i];
                }
            }
        }
    }
}
