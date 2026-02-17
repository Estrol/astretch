use std::sync::Arc;

use num::Complex;
use rustfft::Fft;

/// A simple wrapper around rustfft's FFT.
#[derive(Clone, Default)]
pub struct Pow2FFT<T: super::Sample> {
    size: usize,
    fft: Option<Arc<dyn Fft<T>>>,
    ifft: Option<Arc<dyn Fft<T>>>,
    scratch: Vec<Complex<T>>,
    split_temp: Vec<Complex<T>>,
}

impl<T: super::Sample> std::fmt::Debug for Pow2FFT<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pow2FFT")
            .field("size", &self.size)
            .finish()
    }
}

impl<T: super::Sample> Pow2FFT<T> {
    pub fn new(size: usize) -> Self {
        let mut this = Self::default();
        this.resize(size);

        this
    }

    pub fn resize(&mut self, new_size: usize) {
        if self.size == new_size {
            return;
        }

        if !new_size.is_power_of_two() {
            panic!("Pow2FFT::resize: size must be a power of two");
        }

        self.size = new_size;
        let mut planner = rustfft::FftPlanner::<T>::new();

        let fft = planner.plan_fft_forward(new_size);
        let ifft = planner.plan_fft_inverse(new_size);

        let scratch_len = fft.get_inplace_scratch_len()
            .max(ifft.get_inplace_scratch_len());

        self.fft = Some(Arc::from(fft));
        self.ifft = Some(Arc::from(ifft));
        self.scratch.resize(scratch_len, Complex::default());
        self.split_temp.resize(new_size, Complex::default());
    }

    pub fn fft(&mut self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        if input.len() < self.size || output.len() < self.size {
            panic!("Pow2FFT::fft: input and output sizes must be at least FFT size: expected at least {}, got input {}, output {}", self.size, input.len(), output.len());
        }

        let input = &input[..self.size];
        let output = &mut output[..self.size];

        output.copy_from_slice(input);

        if let Some(fft) = &self.fft {
            fft.process_with_scratch(output, &mut self.scratch);
        }
    }

    pub fn ifft(&mut self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        if input.len() < self.size || output.len() < self.size {
            panic!("Pow2FFT::ifft: input and output sizes must be at least FFT size: expected at least {}, got input {}, output {}", self.size, input.len(), output.len());
        }

        let input = &input[..self.size];
        let output = &mut output[..self.size];

        output.copy_from_slice(input);

        if let Some(ifft) = &self.ifft {
            ifft.process_with_scratch(output, &mut self.scratch);
        }
    }

    pub fn fft_split(&mut self, input_r: &[T], input_i: &[T], output_r: &mut [T], output_i: &mut [T]) {
        if input_r.len() < self.size || input_i.len() < self.size
            || output_r.len() < self.size || output_i.len() < self.size {
            panic!("Pow2FFT::fft_split: input and output sizes must be at least FFT size");
        }

        let input_r = &input_r[..self.size];
        let input_i = &input_i[..self.size];
        let output_r = &mut output_r[..self.size];
        let output_i = &mut output_i[..self.size];

        for i in 0..self.size {
            self.split_temp[i] = Complex {
                re: input_r[i],
                im: input_i[i],
            };
        }

        if let Some(fft) = &self.fft {
            fft.process_with_scratch(&mut self.split_temp, &mut self.scratch);
        }

        for i in 0..self.size {
            output_r[i] = self.split_temp[i].re;
            output_i[i] = self.split_temp[i].im;
        }
    }

    pub fn ifft_split(&mut self, input_r: &[T], input_i: &[T], output_r: &mut [T], output_i: &mut [T]) {
        if input_r.len() < self.size || input_i.len() < self.size 
            || output_r.len() < self.size || output_i.len() < self.size {
            panic!("Pow2FFT::ifft_split: input and output sizes must be at least FFT size");
        }

        let input_r = &input_r[..self.size];
        let input_i = &input_i[..self.size];
        let output_r = &mut output_r[..self.size];
        let output_i = &mut output_i[..self.size];

        for i in 0..self.size {
            self.split_temp[i] = Complex {
                re: input_r[i],
                im: input_i[i],
            };
        }

        if let Some(ifft) = &self.ifft {
            ifft.process_with_scratch(&mut self.split_temp, &mut self.scratch);
        }

        for i in 0..self.size {
            output_r[i] = self.split_temp[i].re;
            output_i[i] = self.split_temp[i].im;
        }
    }
}
