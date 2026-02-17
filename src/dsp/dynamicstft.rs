use alloc::vec::Vec;
use num::complex::Complex;

use super::RealFFT;
use crate::misc;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Spectrum {
    Packed,
    Modified,
    Unpacked,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum WindowShape {
    Ignore,
    Acg,
    Kaiser,
}

const ALMOST_ZERO: f32 = 1e-30;
type FFT<T> = RealFFT<T, false, false>;

// A self-normalising STFT, with variable position/window for output blocks
#[derive(Debug, Clone)]
pub struct DynamicSTFT<T: super::Sample> {
    pub(crate) analysis_channels: usize,
    pub(crate) synthesis_channels: usize,
    pub(crate) input_length_samples: usize,
    pub(crate) block_samples: usize,
    pub(crate) fft_samples: usize,
    pub(crate) fft_bins: usize,
    pub(crate) default_intervals: usize,

    pub(crate) analysis_window: Vec<T>,
    pub(crate) synthesis_window: Vec<T>,
    pub(crate) analysis_offset: usize,
    pub(crate) synthesis_offset: usize,

    pub(crate) spectrum_buffer: Vec<Complex<T>>,
    pub(crate) time_buffer: Vec<T>,

    pub(crate) samples_since_synthesis: usize,
    pub(crate) samples_since_analysis: usize,

    pub(crate) input: Input<T>,
    pub(crate) output: Output<T>,

    pub(crate) split_computation: bool,
    pub(crate) fft: FFT<T>,
    pub(crate) spectrum_type: Spectrum,
}

impl<T: super::Sample> DynamicSTFT<T> {
    pub fn new() -> Self {
        Self {
            analysis_channels: 0,
            synthesis_channels: 0,
            input_length_samples: 0,
            block_samples: 0,
            fft_samples: 0,
            fft_bins: 0,
            default_intervals: 0,

            analysis_window: Vec::new(),
            synthesis_window: Vec::new(),
            analysis_offset: 0,
            synthesis_offset: 0,

            spectrum_buffer: Vec::new(),
            time_buffer: Vec::new(),

            samples_since_synthesis: 0,
            samples_since_analysis: 0,

            input: Input::new(),
            output: Output::new(),

            split_computation: false,
            fft: FFT::new(0),
            spectrum_type: Spectrum::Packed,
        }
    }

    pub fn with_spectrum(spectrum_type: Spectrum) -> Self {
        let mut stft = Self::new();
        stft.spectrum_type = spectrum_type;

        // This add additional CPU cyles,
        // but it ensures that the FFT is configured correctly
        // for the spectrum type from the start.
        stft.fft.set_half_bin_shift(stft.is_modified());
        stft
    }

    pub fn configure(
        &mut self,
        in_channels: usize,
        out_channels: usize,
        block_samples: usize,
        extra_input_history: Option<usize>,
        interval_samples: Option<usize>,
        asymmetry: Option<T>,
    ) {
        self.analysis_channels = in_channels;
        self.synthesis_channels = out_channels;
        self.block_samples = block_samples;
        self.fft_samples = FFT::<T>::fast_size_above((block_samples + 1) / 2) * 2;

        self.fft.resize(self.fft_samples);
        self.fft_bins =
            self.fft_samples / 2 + misc::bool_to_value(self.spectrum_type == Spectrum::Unpacked);

        self.input_length_samples = block_samples + extra_input_history.unwrap_or(0);
        self.input.buffer.resize(
            self.input_length_samples * self.analysis_channels,
            T::zero(),
        );

        self.output
            .buffer
            .resize(self.block_samples * self.synthesis_channels, T::zero());
        self.output
            .window_products
            .resize(self.block_samples, T::zero());

        self.spectrum_buffer.resize(
            self.fft_bins * self.analysis_channels.max(self.synthesis_channels),
            Complex::default(),
        );
        self.time_buffer.resize(self.fft_samples, T::default());

        self.analysis_window
            .resize(self.block_samples, T::default());
        self.synthesis_window
            .resize(self.block_samples, T::default());

        let interval_samples = interval_samples.unwrap_or(self.block_samples / 4);
        self.set_interval(interval_samples, Some(WindowShape::Acg), asymmetry);

        self.reset(None);
    }

    #[inline]
    pub const fn block_samples(&self) -> usize {
        self.block_samples
    }

    #[inline]
    pub const fn fft_samples(&self) -> usize {
        self.fft_samples
    }

    #[inline]
    pub const fn default_intervals(&self) -> usize {
        self.default_intervals
    }

    #[inline]
    pub const fn bands(&self) -> usize {
        self.fft_bins
    }

    #[inline]
    pub const fn analysis_latency(&self) -> usize {
        self.block_samples - self.analysis_offset
    }

    #[inline]
    pub const fn synthesis_latency(&self) -> usize {
        self.synthesis_offset
    }

    #[inline]
    pub const fn latency(&self) -> usize {
        self.analysis_latency() + self.synthesis_latency()
    }

    pub fn bin_to_freq(&self, b: T) -> T {
        let fft_samples_t = T::from_usize(self.fft_samples).unwrap();
        let value = if self.is_modified() {
            b + T::from_f32(0.5).unwrap()
        } else {
            b
        };

        value / fft_samples_t
    }

    pub fn freq_to_bin(&self, f: T) -> T {
        let fft_samples_t = T::from_usize(self.fft_samples).unwrap();
        let value = if self.is_modified() {
            f * fft_samples_t - T::from_f32(0.5).unwrap()
        } else {
            f * fft_samples_t
        };

        value
    }

    #[inline(always)]
    pub fn is_modified(&self) -> bool {
        self.spectrum_type == Spectrum::Modified
    }

    #[inline(always)]
    pub fn is_packed(&self) -> bool {
        self.spectrum_type == Spectrum::Packed
    }

    #[inline(always)]
    pub fn is_unpacked(&self) -> bool {
        self.spectrum_type == Spectrum::Unpacked
    }

    pub fn reset(&mut self, product_weight: Option<T>) {
        self.input.pos = self.block_samples;
        self.output.pos = 0;

        for v in self.input.buffer.iter_mut() {
            *v = T::default();
        }

        for v in self.output.buffer.iter_mut() {
            *v = T::default();
        }

        for v in self.spectrum_buffer.iter_mut() {
            *v = Complex::default();
        }

        for v in self.output.window_products.iter_mut() {
            *v = product_weight.unwrap_or(T::zero());
        }

        self.add_window_product();

        for i in (0..self.block_samples - self.default_intervals).rev() {
            let value = self.output.window_products[i + self.default_intervals];
            self.output.window_products[i] = self.output.window_products[i] + value;
        }

        let product_weight = product_weight.unwrap_or(T::one());
        let almost_zero = T::from_f32(ALMOST_ZERO).unwrap();

        for v in self.output.window_products.iter_mut() {
            *v = *v * product_weight + almost_zero;
        }

        self.move_output(self.default_intervals);
    }

    #[inline(always)]
    pub fn write_input_simple(&mut self, channel: usize, input: &[T]) {
        self.write_input(channel, input.len(), input);
    }

    #[inline(always)]
    pub fn write_input(&mut self, channel: usize, length: usize, input: &[T]) {
        self.write_input_ex(channel, 0, length, input);
    }

    pub fn write_input_ex(&mut self, channel: usize, offset: usize, length: usize, input: &[T]) {
        let buffer = &mut self.input.buffer[(channel * self.input_length_samples)..];

        let offset_pos = (self.input.pos + offset) % self.input_length_samples;
        let input_wrap_index = self.input_length_samples - offset_pos;
        let chunk1 = length.min(input_wrap_index);

        for i in 0..chunk1 {
            let i2 = offset_pos + i;
            buffer[i2] = input[i];
        }

        for i in chunk1..length {
            let i2 = i + offset_pos - self.input_length_samples;
            buffer[i2] = input[i];
        }
    }

    pub fn move_input(&mut self, samples: usize, clear_input: bool) {
        if clear_input {
            let input_wrap_index = self.input_length_samples - self.input.pos;
            let chunk1 = samples.min(input_wrap_index);

            for c in 0..self.analysis_channels {
                let buffer = &mut self.input.buffer;

                for i in 0..chunk1 {
                    buffer[(c * self.input_length_samples) + self.input.pos + i] = T::zero();
                }

                for i in chunk1..samples {
                    buffer[(c * self.input_length_samples) + i + self.input.pos
                        - self.input_length_samples] = T::zero();
                }
            }
        }

        self.input.pos = (self.input.pos + samples) % self.input_length_samples;
        self.samples_since_analysis += samples;
    }

    pub const fn samples_since_analysis(&self) -> usize {
        self.samples_since_analysis
    }

    pub fn finish_output(&mut self, strength: Option<T>, offset: Option<usize>) {
        let mut max_window_output = T::zero();

        let strength = strength.unwrap_or(T::one());
        let offset = offset.unwrap_or(0);

        let chunk1 = offset.max(self.block_samples.min(self.block_samples - self.output.pos));

        for i in offset..chunk1 {
            let i2 = self.output.pos + i;
            let wp = &mut self.output.window_products[i2];

            max_window_output = max_window_output.max(*wp);
            *wp = *wp + ((max_window_output - *wp) * strength);
        }

        for i in chunk1..self.block_samples {
            let i2 = i + self.output.pos - self.block_samples;
            let wp = &mut self.output.window_products[i2];

            max_window_output = max_window_output.max(*wp);
            *wp = *wp + ((max_window_output - *wp) * strength);
        }
    }

    #[inline(always)]
    pub fn read_output_simple(&mut self, channel: usize, output: &mut [T]) {
        self.read_output(channel, output.len(), output);
    }

    #[inline(always)]
    pub fn read_output(&mut self, channel: usize, length: usize, output: &mut [T]) {
        self.read_output_ex(channel, 0, length, output);
    }

    pub fn read_output_ex(
        &mut self,
        channel: usize,
        offset: usize,
        length: usize,
        output: &mut [T],
    ) {
        let buffer = &self.output.buffer[(channel * self.block_samples)..];

        let offset_pos = (self.output.pos + offset) % self.block_samples;
        let output_wrap_index = self.block_samples - offset_pos;
        let chunk1 = length.min(output_wrap_index);

        for i in 0..chunk1 {
            let i2 = offset_pos + i;
            output[i] = buffer[i2] / self.output.window_products[i2];
        }

        for i in chunk1..length {
            let i2 = i + offset_pos - self.block_samples;
            output[i] = buffer[i2] / self.output.window_products[i2];
        }
    }

    #[inline(always)]
    pub fn add_output_simple(&mut self, channel: usize, output: &[T]) {
        self.add_output(channel, output.len(), output);
    }

    #[inline(always)]
    pub fn add_output(&mut self, channel: usize, length: usize, output: &[T]) {
        self.add_output_ex(channel, 0, length, output);
    }

    pub fn add_output_ex(
        &mut self,
        channel: usize,
        offset: usize,
        mut length: usize,
        output: &[T],
    ) {
        length = length.min(self.block_samples);

        // let buffer = &mut self.output.buffer;
        let buffer = &mut self.output.buffer[(channel * self.block_samples)..];

        let offset_pos = (self.output.pos + offset) % self.block_samples;
        let output_wrap_index = self.block_samples - offset_pos;
        let chunk1 = length.min(output_wrap_index);

        for i in 0..chunk1 {
            let i2 = offset_pos + i;
            let value = output[i] * self.output.window_products[i2];
            buffer[i2] = buffer[i2] + value;
        }

        for i in chunk1..length {
            let i2 = i + offset_pos - self.block_samples;
            let value = output[i] * self.output.window_products[i2];
            buffer[i2] = buffer[i2] + value;
        }
    }

    #[inline(always)]
    pub fn replace_output_simple(&mut self, channel: usize, output: &[T]) {
        self.replace_output(channel, output.len(), output);
    }

    #[inline(always)]
    pub fn replace_output(&mut self, channel: usize, length: usize, output: &[T]) {
        self.replace_output_ex(channel, 0, length, output);
    }

    pub fn replace_output_ex(
        &mut self,
        channel: usize,
        offset: usize,
        mut length: usize,
        output: &[T],
    ) {
        length = length.min(self.block_samples);

        let buffer = &mut self.output.buffer;

        let offset_pos = (self.output.pos + offset) % self.block_samples;
        let output_wrap_index = self.block_samples - offset_pos;
        let chunk1 = length.min(output_wrap_index);

        let channel_index = channel * self.block_samples;

        for i in 0..chunk1 {
            let i2 = offset_pos + i;
            buffer[channel_index + i2] = output[i] * self.output.window_products[i2];
        }

        for i in chunk1..length {
            let i2 = i + offset_pos - self.block_samples;
            buffer[channel_index + i2] = output[i] * self.output.window_products[i2];
        }
    }

    pub fn move_output(&mut self, samples: usize) {
        if samples == 1 {
            for c in 0..self.synthesis_channels {
                let i = self.output.pos + c * self.block_samples;
                self.output.buffer[i] = T::zero();
            }

            self.output.window_products[self.output.pos] = T::from_f32(ALMOST_ZERO).unwrap();
            self.output.pos += 1;

            if self.output.pos >= self.block_samples {
                self.output.pos = 0;
            }

            return;
        }

        let output_wrap_index = self.block_samples - self.output.pos;
        let chunk1 = samples.min(output_wrap_index);

        for c in 0..self.synthesis_channels {
            let buffer = &mut self.output.buffer;
            let channel_index = c * self.block_samples;

            for i in 0..chunk1 {
                buffer[channel_index + self.output.pos + i] = T::zero();
            }

            for i in chunk1..samples {
                buffer[channel_index + i + self.output.pos - self.block_samples] = T::zero();
            }
        }

        let almost_zero = T::from_f32(ALMOST_ZERO).unwrap();

        for i in 0..chunk1 {
            let i2 = self.output.pos + i;
            self.output.window_products[i2] = almost_zero;
        }

        for i in chunk1..samples {
            let i2 = i + self.output.pos - self.block_samples;
            self.output.window_products[i2] = almost_zero;
        }

        self.output.pos = (self.output.pos + samples) % self.block_samples;
        self.samples_since_synthesis += samples;
    }

    pub const fn samples_since_synthesis(&self) -> usize {
        self.samples_since_synthesis
    }

    pub fn spectrum(&self, channel: usize) -> &[Complex<T>] {
        &self.spectrum_buffer[(channel * self.fft_bins)..]
    }

    pub fn spectrum_mut(&mut self, channel: usize) -> &mut [Complex<T>] {
        &mut self.spectrum_buffer[(channel * self.fft_bins)..]
    }

    pub fn analysis_window(&self) -> &[T] {
        &self.analysis_window
    }

    pub fn analysis_window_mut(&mut self) -> &mut [T] {
        &mut self.analysis_window
    }

    pub fn set_analysis_offset(&mut self, offset: usize) {
        self.analysis_offset = offset;
    }

    pub const fn analysis_offset(&self) -> usize {
        self.analysis_offset
    }

    pub fn synthesis_window(&self) -> &[T] {
        &self.synthesis_window
    }

    pub fn synthesis_window_mut(&mut self) -> &mut [T] {
        &mut self.synthesis_window
    }

    pub fn set_synthesis_offset(&mut self, offset: usize) {
        self.synthesis_offset = offset;
    }

    pub const fn synthesis_offset(&self) -> usize {
        self.synthesis_offset
    }

    pub fn set_interval(
        &mut self,
        default_interval: usize,
        window_shape: Option<WindowShape>,
        asymmetry: Option<T>,
    ) {
        let window_shape = window_shape.unwrap_or(WindowShape::Ignore);
        let asymmetry = asymmetry.unwrap_or(T::zero());

        self.default_intervals = default_interval;

        match window_shape {
            WindowShape::Acg => {
                let window = ApproximateConfinedGaussian::with_bandwidth(
                    T::from_usize(self.block_samples / default_interval).unwrap(),
                );
                window.fill(
                    &mut self.synthesis_window,
                    self.block_samples,
                    asymmetry,
                    false,
                );
            }
            WindowShape::Kaiser => {
                let kaiser = Kaiser::with_bandwidth(
                    T::from_usize(self.block_samples / default_interval).unwrap(),
                    true,
                );
                kaiser.fill(
                    &mut self.synthesis_window,
                    self.block_samples,
                    asymmetry,
                    true,
                );
            }
            WindowShape::Ignore => return,
        }

        self.synthesis_offset = self.block_samples / 2;
        self.analysis_offset = self.synthesis_offset;

        if self.analysis_channels == 0 {
            for v in self.analysis_window.iter_mut() {
                *v = T::one();
            }
        } else if asymmetry == T::zero() {
            force_perfect_reconstruction(
                &mut self.synthesis_window,
                self.block_samples,
                self.default_intervals,
            );
            for i in 0..self.block_samples {
                self.analysis_window[i] = self.synthesis_window[i];
            }
        } else {
            for i in 0..self.block_samples {
                self.analysis_window[i] = self.synthesis_window[self.block_samples - 1 - i];
            }
        }

        for i in 0..self.block_samples {
            if self.analysis_window[i] > self.analysis_window[self.analysis_offset] {
                self.analysis_offset = i;
            }

            if self.synthesis_window[i] > self.synthesis_window[self.synthesis_offset] {
                self.synthesis_offset = i;
            }
        }
    }

    pub fn swap_input(&mut self, other: &mut Input<T>) {
        self.input.swap(other);
    }

    pub fn swap_output(&mut self, other: &mut Output<T>) {
        self.output.swap(other);
    }

    pub fn analyse(&mut self, samples_in_past: Option<usize>) {
        let samples_in_past = samples_in_past.unwrap_or(0);

        for s in 0..self.analyse_steps() {
            self.analyse_step(s, samples_in_past)
        }
    }

    pub fn analyse_steps(&self) -> usize {
        if self.split_computation {
            let step = self.fft.steps() + 1;
            self.analysis_channels * step
        } else {
            self.analysis_channels
        }
    }

    pub fn analyse_step(&mut self, mut step: usize, samples_in_past: usize) {
        let fft_steps = if self.split_computation {
            self.fft.steps()
        } else {
            0
        };
        let channel = step / (fft_steps + 1);
        step = misc::subtract_non_negative(step, channel * (fft_steps + 1));

        if misc::post_decrement(&mut step) == 0 {
            let offset_pos = (self.input_length_samples * 2 + self.input.pos
                - self.block_samples
                - samples_in_past)
                % self.input_length_samples;
            let input_wrap_index = self.input_length_samples - offset_pos;
            let chunk1 = self.analysis_offset.min(input_wrap_index);
            let chunk2 = self
                .analysis_offset
                .max(self.block_samples.min(input_wrap_index));

            self.samples_since_analysis = samples_in_past;

            let is_modified = self.is_modified();
            let buffer = &self.input.buffer;

            for i in 0..chunk1 {
                let w = if is_modified {
                    -self.analysis_window[i]
                } else {
                    self.analysis_window[i]
                };

                let ti = i + (self.fft_samples - self.analysis_offset);
                let bi = offset_pos + i;

                self.time_buffer[ti] = buffer[(channel * self.input_length_samples) + bi] * w;
            }

            for i in chunk1..self.analysis_offset {
                let w = if is_modified {
                    -self.analysis_window[i]
                } else {
                    self.analysis_window[i]
                };

                let ti = i + (self.fft_samples - self.analysis_offset);
                let bi = i + offset_pos - self.input_length_samples;

                self.time_buffer[ti] = buffer[(channel * self.input_length_samples) + bi] * w;
            }

            for i in self.analysis_offset..chunk2 {
                let w = self.analysis_window[i];
                let ti = i - self.analysis_offset;
                let bi = offset_pos + i;
                self.time_buffer[ti] = buffer[(channel * self.input_length_samples) + bi] * w;
            }

            for i in chunk2..self.block_samples {
                let w = self.analysis_window[i];
                let ti = i - self.analysis_offset;
                let bi = i + offset_pos - self.input_length_samples;
                self.time_buffer[ti] = buffer[(channel * self.input_length_samples) + bi] * w;
            }

            for i in (self.block_samples - self.analysis_offset)
                ..(self.fft_samples - self.analysis_offset)
            {
                self.time_buffer[i] = T::zero();
            }

            if self.split_computation {
                return;
            }
        }

        if self.split_computation {
            self.fft.fft_ex(
                step,
                &self.time_buffer,
                &mut self.spectrum_buffer[(channel * self.fft_bins)..],
            );

            if self.is_unpacked() && step == self.fft.steps() - 1 {
                let fft_bin = self.fft_bins - 1;

                let spectrum = self.spectrum_mut(channel);
                spectrum[fft_bin] = Complex::new(spectrum[0].im, T::zero());
                spectrum[0].im = T::zero();
            }
        } else {
            self.fft.fft(
                &self.time_buffer,
                &mut self.spectrum_buffer[(channel * self.fft_bins)..],
            );

            if self.is_unpacked() {
                let fft_bin = self.fft_bins - 1;

                let spectrum = self.spectrum_mut(channel);
                spectrum[fft_bin] = Complex::new(spectrum[0].im, T::zero());
                spectrum[0].im = T::zero();
            }
        }
    }

    pub fn synthesis(&mut self) {
        for s in 0..self.synthesis_steps() {
            self.synthesis_step(s);
        }
    }

    pub fn synthesis_steps(&self) -> usize {
        if self.split_computation {
            self.synthesis_channels * (self.fft.steps() + 1) + 1
        } else {
            self.synthesis_channels
        }
    }

    pub fn synthesis_step(&mut self, mut step: usize) {
        if step == 0 {
            self.add_window_product();

            if self.split_computation {
                return;
            }
        }

        if self.split_computation {
            step = misc::subtract_non_negative(step, 1);
        }

        let fft_steps = if self.split_computation {
            self.fft.steps()
        } else {
            0
        };
        let channel = step / (fft_steps + 1);
        step = misc::subtract_non_negative(step, channel * (fft_steps + 1));

        {
            // Repack spectrum
            let fft_bins = self.fft_bins - 1;
            let is_unpacked = self.is_unpacked();

            let spectrum = self.spectrum_mut(channel);

            if is_unpacked && step == 0 {
                spectrum[0] = Complex {
                    re: spectrum[0].re,
                    im: spectrum[fft_bins].re,
                };
            }
        }

        if self.split_computation {
            if step < fft_steps {
                let spectrum = &self.spectrum_buffer[(channel * self.fft_bins)..];

                self.fft.ifft_ex(step, spectrum, &mut self.time_buffer);

                return;
            }
        } else {
            let spectrum = &self.spectrum_buffer[(channel * self.fft_bins)..];

            self.fft.ifft(spectrum, &mut self.time_buffer);
        }

        let is_modified = self.is_modified();
        let buffer = &mut self.output.buffer;

        let output_wrap_index = self.block_samples - self.output.pos;
        let chunk1 = self.synthesis_offset.min(output_wrap_index);
        let chunk2 = self
            .block_samples
            .min(self.synthesis_offset.max(output_wrap_index));
        let channel_index = channel * self.block_samples;

        for i in 0..chunk1 {
            let w = if is_modified {
                -self.synthesis_window[i]
            } else {
                self.synthesis_window[i]
            };

            let ti = i + (self.fft_samples - self.synthesis_offset);
            let bi = self.output.pos + i;

            let value = self.time_buffer[ti] * w;
            buffer[channel_index + bi] = buffer[channel_index + bi] + value;
        }

        for i in chunk1..self.synthesis_offset {
            let w = if is_modified {
                -self.synthesis_window[i]
            } else {
                self.synthesis_window[i]
            };

            let ti = i + (self.fft_samples - self.synthesis_offset);
            let bi = i + self.output.pos - self.block_samples;

            let value = self.time_buffer[ti] * w;
            buffer[channel_index + bi] = buffer[channel_index + bi] + value;
        }

        for i in self.synthesis_offset..chunk2 {
            let w = self.synthesis_window[i];
            let ti = i - self.synthesis_offset;
            let bi = self.output.pos + i;

            let value = self.time_buffer[ti] * w;
            buffer[channel_index + bi] = buffer[channel_index + bi] + value;
        }

        for i in chunk2..self.block_samples {
            let w = self.synthesis_window[i];
            let ti = i - self.synthesis_offset;
            let bi = i + self.output.pos - self.block_samples;

            let value = self.time_buffer[ti] * w;
            buffer[channel_index + bi] = buffer[channel_index + bi] + value;
        }
    }

    pub fn add_window_product(&mut self) {
        self.samples_since_synthesis = 0;

        let window_shift = self.synthesis_offset - self.analysis_offset;
        let w_min = window_shift.max(0);
        let w_max = self.block_samples.min(self.block_samples + window_shift);

        let window_product = &mut self.output.window_products;
        let output_wrap_index = self.block_samples - self.output.pos;
        let chuck1 = w_max.min(w_min.max(output_wrap_index));

        for i in w_min..chuck1 {
            let wa = self.analysis_window[i - window_shift];
            let ws = self.synthesis_window[i];
            let bi = self.output.pos + i;

            let value = wa * ws * T::from_usize(self.fft_samples).unwrap();
            window_product[bi] = window_product[bi] + value;
        }

        for i in chuck1..w_max {
            let wa = self.analysis_window[i - window_shift];
            let ws = self.synthesis_window[i];
            let bi = i + self.output.pos - self.block_samples;

            let value = wa * ws * T::from_usize(self.fft_samples).unwrap();
            window_product[bi] = window_product[bi] + value;
        }
    }
}

#[derive(Debug, Clone)]
pub struct Input<T: super::Sample> {
    pub buffer: Vec<T>,
    pub pos: usize,
}

impl<T: super::Sample> Input<T> {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            pos: 0,
        }
    }

    pub fn swap(&mut self, other: &mut Self) {
        core::mem::swap(&mut self.buffer, &mut other.buffer);
        core::mem::swap(&mut self.pos, &mut other.pos);
    }
}

#[derive(Debug, Clone)]
pub struct Output<T: super::Sample> {
    pub buffer: Vec<T>,
    pub window_products: Vec<T>,
    pub pos: usize,
}

impl<T: super::Sample> Output<T> {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            window_products: Vec::new(),
            pos: 0,
        }
    }

    pub fn swap(&mut self, other: &mut Self) {
        core::mem::swap(&mut self.buffer, &mut other.buffer);
        core::mem::swap(&mut self.window_products, &mut other.window_products);
        core::mem::swap(&mut self.pos, &mut other.pos);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Kaiser<T: super::Sample> {
    pub beta: T,
    pub inv_b0: T,
}

impl<T: super::Sample> Kaiser<T> {
    pub fn new(beta: T) -> Self {
        let b0 = Self::bessel_i0(beta);
        Self {
            beta,
            inv_b0: T::one() / b0,
        }
    }

    pub fn with_bandwidth(bandwidth: T, heuristic_optiomal: bool) -> Self {
        let bandwidth = Self::bandwidth_to_beta(bandwidth, heuristic_optiomal);

        return Self::new(bandwidth);
    }

    pub fn fill(&self, data: &mut [T], size: usize, warp: T, for_synthesis: bool) {
        let inv_size = T::one() / T::from_usize(size).unwrap();

        let offset_i = if size & 1 == 0 {
            1
        } else if for_synthesis {
            0
        } else {
            1
        };

        let scale = T::from_f32(2.0).unwrap() * inv_size * T::from_usize(offset_i).unwrap();

        for i in 0..size {
            let r0 = scale * T::from_usize(i).unwrap() - T::one();
            let r = (r0 + warp) / (T::one() + r0 * warp);
            let arg = (T::one() - r * r).sqrt();
            data[i] = Self::bessel_i0(self.beta * arg) * self.inv_b0;
        }
    }

    pub fn bessel_i0(x: T) -> T {
        let mut sum = T::one();
        let mut u = T::one();
        let half_x = x / T::from_f32(2.0).unwrap();
        let mut n = T::one();

        loop {
            u = u * half_x * half_x / (n * n);
            sum = sum + u;

            if u < T::from_f32(1e-8).unwrap() * sum {
                break;
            }

            n = n + T::one();
        }

        sum
    }

    pub fn heuristic_bandwidth(bandwidth: T) -> T {
        let three = T::from_f32(3.0).unwrap();
        let eight = T::from_f32(8.0).unwrap();
        let quarter = T::from_f32(0.25).unwrap();

        bandwidth
            + eight / ((bandwidth + three) * (bandwidth + three))
            + quarter * (three - bandwidth).max(T::zero())
    }

    pub fn bandwidth_to_beta(mut bandwidth: T, heuristic_optiomal: bool) -> T {
        if heuristic_optiomal {
            bandwidth = Self::heuristic_bandwidth(bandwidth);
        }

        bandwidth = bandwidth.max(T::from_f32(2.0).unwrap());
        let alpha = (bandwidth * bandwidth * T::from_f32(0.25).unwrap() - T::one()).sqrt();

        alpha * T::from_f32(core::f32::consts::PI).unwrap()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ApproximateConfinedGaussian<T: super::Sample> {
    pub gaussian_factor: T,
}

impl<T: super::Sample> ApproximateConfinedGaussian<T> {
    pub fn new(sigma: T) -> Self {
        Self {
            gaussian_factor: T::from_f32(0.0625).unwrap() / (sigma * sigma),
        }
    }

    pub fn with_bandwidth(bandwidth: T) -> Self {
        let sigma = Self::bandwidth_to_sigma(bandwidth);

        Self::new(sigma)
    }

    pub fn gaussian(&self, x: T) -> T {
        (-self.gaussian_factor * x * x).exp()
    }

    pub fn bandwidth_to_sigma(bandwidth: T) -> T {
        T::from_f32(0.3).unwrap() / bandwidth.sqrt()
    }

    pub fn fill(&self, data: &mut [T], size: usize, warp: T, for_synthesis: bool) {
        let inv_size = T::one() / T::from_usize(size).unwrap();
        let offset_scale = self.gaussian(T::one())
            / (self.gaussian(T::from_f32(3.0).unwrap())
                + self.gaussian(T::from_f32(-1.0).unwrap()));
        let norm = T::one()
            / (self.gaussian(T::zero())
                - T::from_f32(2.0).unwrap()
                    * offset_scale
                    * self.gaussian(T::from_f32(2.0).unwrap()));
        let offset_i = if size & 1 != 0 {
            1
        } else {
            if for_synthesis { 0 } else { 2 }
        };

        for i in 0..size {
            let mut r = (T::from_usize(2 * i + offset_i).unwrap() * inv_size) - T::one();
            r = (r + warp) / (T::one() + r * warp);

            let v = norm
                * (self.gaussian(r)
                    - offset_scale
                        * (self.gaussian(r - T::from_f32(2.0).unwrap())
                            + self.gaussian(r + T::from_f32(2.0).unwrap())));
            data[i] = v;
        }
    }
}

pub(crate) fn force_perfect_reconstruction<T: super::Sample>(
    data: &mut [T],
    window_length: usize,
    interval: usize,
) {
    for i in 0..interval {
        let mut sum2 = T::zero();
        for index in (i..window_length).step_by(interval) {
            let v = data[index];
            sum2 = sum2 + (v * v);
        }

        let factor = T::one() / sum2.sqrt();
        for index in (i..window_length).step_by(interval) {
            data[index] = data[index] * factor;
        }
    }
}
