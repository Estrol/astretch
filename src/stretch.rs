use num::{Complex, Zero};
use rand::SeedableRng;

use crate::dsp::Sample;

use super::{
    dsp::dynamicstft::{
        DynamicSTFT, Input, Output, WindowShape,
    },
    misc
};

pub(crate) const NOISE_FLOOR: f32 = 1e-15f32;
pub(crate) const MAX_CLEAN_STRETCH: f32 = 2.0f32;

pub struct Stretch<T: Sample> {
    pub(crate) split_computation: bool,
    pub(crate) block_process: BlockProcess<T>,
    pub(crate) silent_encounter: usize,
    pub(crate) silent_first: bool,
    pub(crate) freq_multiplier: T,
    pub(crate) freq_tonality_limit: T,
    pub(crate) custom_freq_map: Option<Box<dyn Fn(T) -> T>>,
    pub(crate) formant_compensation: bool,
    pub(crate) formant_multiplier: T,
    pub(crate) inv_formant_multiplier: T,
    pub(crate) stft: DynamicSTFT<T>,
    pub(crate) stashed_input: Input<T>,
    pub(crate) stashed_output: Output<T>,
    pub(crate) temp_process_buffer: Vec<T>,
    pub(crate) temp_pre_roll_buffer: Vec<T>,
    pub(crate) channels: i32,
    pub(crate) bands: i32,
    pub(crate) prev_input_offset: i32,
    pub(crate) seek_time_factor: T,
    pub(crate) channel_bands: StretchBands<T>,
    pub(crate) channel_predictions: StretchPredictions<T>,
    pub(crate) peaks: Vec<Peak<T>>,
    pub(crate) energy: Vec<T>,
    pub(crate) smoothed_energy: Vec<T>,
    pub(crate) output_map: Vec<PitchMapPoint<T>>,
    pub(crate) process_spectrum_steps: usize,
    pub(crate) smooth_energy_state: T,
    pub(crate) freq_estimate_weighted: T,
    pub(crate) freq_estimate_weight: T,
    pub(crate) freq_estimate: T,
    pub(crate) formant_metrics: Vec<T>,
    pub(crate) formant_base_freq: T,
    pub(crate) did_seek: bool,
    pub(crate) random_engine: RandomEngine,
}

// Manually implement clone and debug because custom_freq_map cannot be cloned or printed
impl<T: Sample> Clone for Stretch<T> {
    fn clone(&self) -> Self {
        Self {
            split_computation: self.split_computation,
            block_process: self.block_process.clone(),
            silent_encounter: self.silent_encounter,
            silent_first: self.silent_first,
            freq_multiplier: self.freq_multiplier,
            freq_tonality_limit: self.freq_tonality_limit,
            custom_freq_map: None, // cannot clone closures
            formant_compensation: self.formant_compensation,
            formant_multiplier: self.formant_multiplier,
            inv_formant_multiplier: self.inv_formant_multiplier,
            stft: self.stft.clone(),
            stashed_input: self.stashed_input.clone(),
            stashed_output: self.stashed_output.clone(),
            temp_process_buffer: self.temp_process_buffer.clone(),
            temp_pre_roll_buffer: self.temp_pre_roll_buffer.clone(),
            channels: self.channels,
            bands: self.bands,
            prev_input_offset: self.prev_input_offset,
            seek_time_factor: self.seek_time_factor,
            channel_bands: self.channel_bands.clone(),
            channel_predictions: self.channel_predictions.clone(),
            peaks: self.peaks.clone(),
            energy: self.energy.clone(),
            smoothed_energy: self.smoothed_energy.clone(),
            output_map: self.output_map.clone(),
            process_spectrum_steps: self.process_spectrum_steps,
            smooth_energy_state: self.smooth_energy_state,
            freq_estimate_weighted: self.freq_estimate_weighted,
            freq_estimate_weight: self.freq_estimate_weight,
            freq_estimate: self.freq_estimate,
            formant_metrics: self.formant_metrics.clone(),
            formant_base_freq: self.formant_base_freq,
            did_seek: self.did_seek,
            random_engine: self.random_engine.clone(),
        }
    }
}

impl<T: Sample> std::fmt::Debug for Stretch<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stretch")
            .field("split_computation", &self.split_computation)
            .field("block_process", &self.block_process)
            .field("silent_encounter", &self.silent_encounter)
            .field("silent_first", &self.silent_first)
            .field("freq_multiplier", &self.freq_multiplier)
            .field("freq_tonality_limit", &self.freq_tonality_limit)
            .field("custom_freq_map", &self.custom_freq_map.is_some())
            .field("formant_compensation", &self.formant_compensation)
            .field("formant_multiplier", &self.formant_multiplier)
            .field("inv_formant_multiplier", &self.inv_formant_multiplier)
            .field("stft", &self.stft)
            .field("stashed_input", &self.stashed_input)
            .field("stashed_output", &self.stashed_output)
            .field("temp_process_buffer", &self.temp_process_buffer)
            .field("temp_pre_roll_buffer", &self.temp_pre_roll_buffer)
            .field("channels", &self.channels)
            .field("bands", &self.bands)
            .field("prev_input_offset", &self.prev_input_offset)
            .field("seek_time_factor", &self.seek_time_factor)
            .field("channel_bands", &self.channel_bands)
            .field("channel_predictions", &self.channel_predictions)
            .field("peaks", &self.peaks)
            .field("energy", &self.energy)
            .field("smoothed_energy", &self.smoothed_energy)
            .field("output_map", &self.output_map)
            .field("process_spectrum_steps", &self.process_spectrum_steps)
            .field("smooth_energy_state", &self.smooth_energy_state)
            .field("freq_estimate_weighted", &self.freq_estimate_weighted)
            .field("freq_estimate_weight", &self.freq_estimate_weight)
            .field("freq_estimate", &self.freq_estimate)
            .field("formant_metrics", &self.formant_metrics)
            .field("formant_base_freq", &self.formant_base_freq)
            .field("did_seek", &self.did_seek)
            .field("random_engine", &self.random_engine)
            .finish()
    }
}

impl<T: Sample> Default for Stretch<T> {
    fn default() -> Self {
        Self {
            split_computation: false,
            block_process: BlockProcess::default(),
            silent_encounter: 0,
            silent_first: true,
            freq_multiplier: T::zero(),
            freq_tonality_limit: T::from_f32(0.5).unwrap(),
            custom_freq_map: None,
            formant_compensation: false,
            formant_multiplier: T::one(),
            inv_formant_multiplier: T::one(),
            stft: DynamicSTFT::new(),
            stashed_input: Input::new(),
            stashed_output: Output::new(),
            temp_process_buffer: vec![],
            temp_pre_roll_buffer: vec![],
            channels: 0,
            bands: 0,
            prev_input_offset: -1,
            seek_time_factor: T::one(),
            channel_bands: StretchBands::new(0, 0),
            channel_predictions: StretchPredictions::new(0, 0),
            peaks: vec![],
            energy: vec![],
            smoothed_energy: vec![],
            output_map: vec![],
            process_spectrum_steps: 0,
            smooth_energy_state: T::zero(),
            freq_estimate_weighted: T::zero(),
            freq_estimate_weight: T::zero(),
            freq_estimate: T::zero(),
            formant_metrics: vec![],
            formant_base_freq: T::zero(),
            did_seek: false,
            random_engine: RandomEngine::default(),
        }
    }
}

impl<T: Sample> Stretch<T> {
    /// Creates a new Stretch instance with a random seed.
    pub fn new() -> Self {
        let mut stretch = Stretch::default();

        stretch.random_engine = RandomEngine::new();
        stretch
    }

    /// Creates a new Stretch instance with a specific seed for reproducibility.
    pub fn from_seed(seed: u32) -> Self {
        let mut stretch = Stretch::default();

        stretch.random_engine = RandomEngine::from_seed(seed as u64);
        stretch
    }

    /// Get input latency.
    pub fn input_latency(&self) -> i32 {
        self.stft.analysis_latency() as i32
    }

    /// Get output latency.
    pub fn output_latency(&self) -> i32 {
        self.stft.synthesis_latency() as i32
            + misc::bool_to_value(self.split_computation) as i32
                * self.stft.default_intervals() as i32
    }

    /// Reset the internal state of the stretch. 
    /// This is useful when seeking or when you want to clear any history.
    pub fn reset(&mut self) {
        self.stft.reset(T::from_f32(0.1));

        self.stashed_input = self.stft.input.clone();
        self.stashed_output = self.stft.output.clone();

        self.channel_bands.reset();
        self.silent_encounter = 0;
        self.prev_input_offset = -1;
        self.did_seek = false;
        self.block_process = BlockProcess::default();
        self.freq_estimate_weighted = T::zero();
        self.freq_estimate_weight = T::zero();
    }

    /// A preset configuration for general use. This is a good starting point for most cases.
    pub fn preset_default(&mut self, channels: i32, sample_rate: T, split_computation: bool) {
        let block_samples = (sample_rate.to_f32().unwrap() * 0.12) as i32;
        let interval_samples = (sample_rate.to_f32().unwrap() * 0.03) as i32;

        self.configure(
            channels,
            block_samples,
            interval_samples,
            split_computation,
        );
    }

    /// A cheaper preset configuration that uses less CPU at the cost of some quality. 
    /// This can be useful for real-time applications or when processing power is limited.
    pub fn preset_cheaper(&mut self, channels: i32, sample_rate: T, split_computation: bool) {
        let block_samples = (sample_rate.to_f32().unwrap() * 0.1) as i32;
        let interval_samples = (sample_rate.to_f32().unwrap() * 0.04) as i32;
        
        self.configure(
            channels,
            block_samples,
            interval_samples,
            split_computation,
        );
    }

    /// Configure the stretch with custom parameters. This gives you full control over the behavior of the stretch.
    /// Useful for something like processing very small or very large audio files where you want to optimize for quality or speed.
    pub fn configure(
        &mut self,
        channels: i32,
        block_samples: i32,
        interval_samples: i32,
        split_computation: bool,
    ) {
        self.split_computation = split_computation;
        self.channels = channels;

        self.stft.configure(
            channels as usize,
            channels as usize,
            block_samples as usize,
            Some(interval_samples as usize + 1),
            None,
            None,
        );
        self.stft
            .set_interval(interval_samples as usize, Some(WindowShape::Kaiser), None);
        self.stft.reset(T::from_f32(0.1));

        self.stashed_input = self.stft.input.clone();
        self.stashed_output = self.stft.output.clone();

        let bands = self.stft.bands();

        self.channel_bands.resize(self.channels as usize, bands);
        self.channel_bands.reset();
        self.bands = bands as i32;

        self.peaks.resize(self.bands as usize / 2, Peak::default());
        self.energy.resize(self.bands as usize, T::zero());
        self.smoothed_energy.resize(self.bands as usize, T::zero());
        self.output_map
            .resize(self.bands as usize, PitchMapPoint::default());
        self.channel_predictions
            .resize(self.channels as usize, bands);

        self.block_process = BlockProcess::default();
        self.formant_metrics.resize(self.bands as usize + 2, T::zero());

        self.temp_pre_roll_buffer
            .resize(self.output_latency() as usize * channels as usize, T::zero());
        self.temp_process_buffer
            .resize((block_samples + interval_samples) as usize, T::zero());
    }

    /// Get number of block samples.
    pub fn block_samples(&self) -> i32 {
        self.stft.block_samples() as i32
    }

    /// Get number of interval samples.
    pub fn interval_samples(&self) -> i32 {
        self.stft.default_intervals() as i32
    }

    /// Check whatever the stretch is configured to split the processing.
    /// 
    /// When enabled, the latency will one interval higher.
    pub fn split_computation(&self) -> bool {
        self.split_computation
    }

    pub fn set_transpose_factor(&mut self, multiplier: T, tonality_limit: Option<T>) {
        self.freq_multiplier = multiplier;

        if let Some(limit) = tonality_limit
            && limit > T::zero()
        {
            self.freq_tonality_limit = limit / multiplier.sqrt();
        } else {
            self.freq_tonality_limit = T::one();
        }

        self.custom_freq_map = None;
    }

    pub fn set_transpose_semitones(&mut self, semitones: T, tonality_limit: Option<T>) {
        let multiplier = T::from_f32(
            2.0f32.powf(semitones.to_f32().unwrap() / 12.0)
        ).unwrap();

        self.set_transpose_factor(multiplier, tonality_limit);
    }

    pub fn set_freq_map<F>(&mut self, map: F)
    where
        F: Fn(T) -> T + 'static,
    {
        self.custom_freq_map = Some(Box::new(map));
    }
    
    pub fn set_formant_factor(&mut self, multiplier: T, compesate_pitch: bool) {
        self.formant_multiplier = multiplier;
        self.inv_formant_multiplier = T::one() / multiplier;
        self.formant_compensation = compesate_pitch;
    }

    pub fn set_formant_semitones(&mut self, semitones: T, compesate_pitch: bool) {
        let multiplier = T::from_f32(
            2.0f32.powf(semitones.to_f32().unwrap() / 12.0)
        ).unwrap();

        self.set_formant_factor(multiplier, compesate_pitch);
    }

    pub fn set_formant_base(&mut self, freq: T) {
        self.formant_base_freq = freq;
    }

    pub fn seek<I>(&mut self, input: I, playback_rate: T)
    where
        I: AsRef<[T]>,
    {
        let length = input.as_ref().len() / self.channels as usize;
        self.seek_ex(input, length, playback_rate);
    }

    pub fn seek_ex<I>(&mut self, input: I, input_samples: usize, playback_rate: T)
    where
        I: AsRef<[T]>,
    {
        let inputs = input.as_ref();

        self.temp_process_buffer.resize(0, T::zero());

        let length = self.stft.block_samples() + self.stft.default_intervals();
        self.temp_process_buffer.resize(length, T::zero());

        let index_start = (input_samples as isize - self.temp_process_buffer.len() as isize).max(0);
        let pad_start = (self.temp_process_buffer.len() + index_start as usize) - input_samples;

        let mut total_energy = T::zero();
        for c in 0..self.channels as usize {
            for i in (index_start as usize)..input_samples {
                let s = inputs[i * self.channels as usize + c];
                total_energy = total_energy + (s * s);
                self.temp_process_buffer[i - index_start as usize + pad_start] = s;
            }

            self.stft
                .write_input(c, self.temp_process_buffer.len(), &self.temp_process_buffer);
        }

        self.stft.move_input(self.temp_process_buffer.len(), false);

        if total_energy >= T::from_f32(NOISE_FLOOR).unwrap() {
            self.silent_encounter = 0;
            self.silent_first = true;
        }

        self.did_seek = true;
        self.seek_time_factor = if playback_rate * T::from_usize(self.stft.default_intervals()).unwrap() > T::one() {
            T::one() / playback_rate
        } else {
            T::from_usize(self.stft.default_intervals()).unwrap()
        };
    }

    /// Get minimum number of samples required for seeking.
    pub fn seek_length(&self) -> usize {
        (self.stft.block_samples() + self.stft.default_intervals()) as usize
    }

    pub fn output_seek<I>(&mut self, input: &I, input_samples: usize)
    where
        I: AsRef<[T]>,
    {
        self.reset();

        let surplus_input = (input_samples - self.input_latency() as usize).max(0);
        let playback_rate = T::from_usize(surplus_input).unwrap() / T::from_i32(self.output_latency()).unwrap();

        let seek_samples = input_samples - surplus_input;
        self.seek_ex(input, seek_samples, playback_rate);

        self.temp_pre_roll_buffer
            .resize((self.output_latency() * self.channels) as usize, T::zero());

        let input = input.as_ref();
        let mut temp_pre_roll_buffer = std::mem::take(&mut self.temp_pre_roll_buffer);
        let output_latency = self.output_latency() as usize;

        self.process_internal(
            Some(&input[seek_samples * self.channels as usize..]),
            surplus_input,
            &mut temp_pre_roll_buffer,
            output_latency,
        );

        let mut temp_buffer = vec![T::zero(); output_latency];
        for c in 0..self.channels as usize {
            for i in 0..output_latency {
                temp_buffer[i] = temp_pre_roll_buffer[i * self.channels as usize + c];
                temp_buffer[i] = -temp_buffer[i];
            }

            temp_buffer.reverse();
            self.stft.add_output(c, temp_buffer.len(), &temp_buffer);
        }

        self.temp_pre_roll_buffer = temp_pre_roll_buffer;
    }

    /// Get minimum number of samples required for output seeking.
    pub fn output_seek_length(&self, playback_rate: T) -> i32 {
        self.input_latency() + (playback_rate * T::from_i32(self.output_latency()).unwrap()).to_i32().unwrap()
    }

    pub(crate) fn copy_input<I>(
        &mut self,
        input: Option<&I>,
        to_index: usize,
        prev_copied_input: &mut usize,
    ) where
        I: AsRef<[T]> + ?Sized,
    {
        let length = (self.stft.block_samples() + self.stft.default_intervals())
            .min(to_index - *prev_copied_input);
        self.temp_process_buffer.resize(length, T::zero());

        let offset = to_index - length;
        if let Some(input) = input.as_ref() {
            let inputs = input.as_ref();

            for c in 0..self.channels as usize {
                let offset = offset * self.channels as usize + c;
                for i in 0..length {
                    self.temp_process_buffer[i] = inputs[offset + i * self.channels as usize];
                }

                self.stft.write_input(c, length, &self.temp_process_buffer);
            }
        } else {
            for c in 0..self.channels as usize {
                self.temp_process_buffer.fill(T::zero());
                self.stft.write_input(c, length, &self.temp_process_buffer);
            }
        }

        self.stft.move_input(length, false);
        *prev_copied_input = to_index;
    }

    pub(crate) fn process_internal<I, O>(
        &mut self,
        input: Option<&I>,
        input_sample: usize,
        output: &mut O,
        output_sample: usize,
    ) where
        I: AsRef<[T]> + ?Sized,
        O: AsMut<[T]> + ?Sized,
    {
        let mut prev_copied_input = 0;
        let mut total_energy = T::zero();

        if let Some(input) = input.as_ref() {
            let inputs = input.as_ref();

            for c in 0..self.channels as usize {
                for i in 0..input_sample {
                    let s = inputs[i * self.channels as usize + c];
                    total_energy = total_energy + (s * s);
                }
            }
        }

        let noise_floor = T::from_f32(NOISE_FLOOR).unwrap();
        if total_energy < noise_floor {
            if self.silent_encounter >= 2 * self.stft.block_samples() {
                if self.silent_first {
                    self.silent_first = false;
                    self.block_process = BlockProcess::default();
                    self.channel_bands.reset();
                }

                if input_sample > 0
                    && let Some(input) = input.as_ref()
                {
                    let input = input.as_ref();
                    let output = output.as_mut();

                    // Read and write in interleaved manner
                    for output_index in 0..output_sample {
                        let input_index = output_index % input_sample;

                        for c in 0..self.channels as usize {
                            output[output_index * self.channels as usize + c] =
                                input[input_index * self.channels as usize + c];
                        }
                    }
                } else {
                    let output = output.as_mut();

                    for c in 0..self.channels as usize {
                        for i in 0..output_sample {
                            output[i * self.channels as usize + c] = T::zero();
                        }
                    }
                }

                self.copy_input(input, input_sample, &mut prev_copied_input);
                return;
            } else {
                self.silent_encounter += input_sample;
            }
        } else {
            self.silent_encounter = 0;
            self.silent_first = true;
        }

        for output_index in 0..output_sample {
            let new_block = self.block_process.samples_since_last >= self.stft.default_intervals();

            if new_block {
                self.block_process.step = 0;
                self.block_process.steps = 0;
                self.block_process.samples_since_last = 0;

                let input_offset = (output_index as f32 * input_sample as f32
                    / output_sample as f32)
                    .round() as i32;

                let input_interval = input_offset - self.prev_input_offset;
                self.prev_input_offset = input_offset;

                self.copy_input(
                    input,
                    input_offset as usize,
                    &mut prev_copied_input,
                );

                self.stashed_input = self.stft.input.clone();

                if self.split_computation {
                    self.stashed_output = self.stft.output.clone();

                    let default_interval = self.stft.default_intervals();
                    self.stft.move_output(default_interval);
                }

                self.block_process.new_spectrum = self.did_seek || input_interval > 0;
                self.block_process.mapped_frequencies =
                    self.custom_freq_map.is_some() || self.freq_multiplier != T::one();

                if self.block_process.new_spectrum {
                    self.block_process.reanalyze_prev = self.did_seek
                        || (input_interval as i32 - self.stft.default_intervals() as i32).abs() > 1;
                    if self.block_process.reanalyze_prev {
                        self.block_process.steps += self.stft.analyse_steps() + 1;
                    }

                    self.block_process.steps += self.stft.analyse_steps() + 1;
                }

                self.block_process.process_formanets = self.formant_multiplier != T::one()
                    || (self.formant_compensation && self.block_process.mapped_frequencies);
                self.block_process.time_factor = if self.did_seek {
                    self.seek_time_factor
                } else {
                    T::from_usize(self.stft.default_intervals()).unwrap() / T::from_i32(input_interval.max(1)).unwrap()
                };

                self.did_seek = false;

                self.update_process_spectrum_steps();

                self.block_process.steps += self.process_spectrum_steps;
                self.block_process.steps += self.stft.synthesis_steps() + 1;
            }

            let mut process_to_step = if new_block {
                self.block_process.steps
            } else {
                0
            };

            if self.split_computation {
                let process_ratio = (T::from_usize(self.block_process.samples_since_last).unwrap() + T::one())
                    / T::from_usize(self.stft.default_intervals()).unwrap();

                let target = (T::from_usize(self.block_process.steps).unwrap() * process_ratio).ceil().to_usize().unwrap();
                process_to_step = self.block_process.steps.min(target);
            }

            while self.block_process.step < process_to_step {
                let mut step = misc::post_increment(&mut self.block_process.step, usize::MAX);

                if self.block_process.new_spectrum {
                    if self.block_process.reanalyze_prev {
                        if step < self.stft.analyse_steps() {
                            self.stft.swap_input(&mut self.stashed_input);
                            self.stft.analyse_step(step, self.stft.default_intervals());
                            self.stft.swap_input(&mut self.stashed_input);
                            continue;
                        }

                        step = misc::subtract_non_negative(step, self.stft.analyse_steps());

                        if step < 1 {
                            for c in 0..self.channels as usize {
                                let channel_bands = self.channel_bands.bands_for_channel_mut(c);
                                let spectrum_bands = self.stft.spectrum(c);

                                for b in 0..self.bands as usize {
                                    channel_bands[b].prev_input = spectrum_bands[b];
                                }
                            }
                            continue;
                        }

                        step = misc::subtract_non_negative(step, 1);
                    }

                    // Analyse latest stashed input
                    if step < self.stft.analyse_steps() {
                        self.stft.swap_input(&mut self.stashed_input);
                        self.stft.analyse_step(step, 0);
                        self.stft.swap_input(&mut self.stashed_input);
                        continue;
                    }

                    step = misc::subtract_non_negative(step, self.stft.analyse_steps());

                    if step < 1 {
                        // Copy analysed spectrum to channel bands
                        for c in 0..self.channels as usize {
                            let channel_bands = self.channel_bands.bands_for_channel_mut(c);
                            let spectrum_bands = self.stft.spectrum(c);

                            for b in 0..self.bands as usize {
                                channel_bands[b].input = spectrum_bands[b];
                            }
                        }
                        continue;
                    }

                    step = misc::subtract_non_negative(step, 1);
                }

                if step < self.process_spectrum_steps {
                    self.process_spectrum(step);
                    continue;
                }

                step = misc::subtract_non_negative(step, self.process_spectrum_steps);

                if step < 1 {
                    // Copy band object to spectrum
                    for c in 0..self.channels as usize {
                        let channel_bands = self.channel_bands.bands_for_channel(c);
                        let spectrum_bands = self.stft.spectrum_mut(c);

                        for b in 0..self.bands as usize {
                            spectrum_bands[b] = channel_bands[b].output;
                        }
                    }
                    continue;
                }

                step = misc::subtract_non_negative(step, 1);

                if step < self.stft.synthesis_steps() {
                    self.stft.synthesis_step(step);
                    continue;
                }
            }

            misc::pre_increment(&mut self.block_process.samples_since_last, usize::MAX);

            if self.split_computation {
                self.stft.swap_output(&mut self.stashed_output);
            }

            let output = output.as_mut();

            // write output in interleaved manner
            for c in 0..self.channels as usize {
                let mut v = [T::zero(); 1];
                self.stft.read_output(c, 1, &mut v);

                output[output_index * self.channels as usize + c] = v[0];
            }

            self.stft.move_output(1);

            if self.split_computation {
                self.stft.swap_output(&mut self.stashed_output);
            }
        }

        self.copy_input(input, input_sample, &mut prev_copied_input);
        self.prev_input_offset -= input_sample as i32;
    }

    pub fn process<I, O>(
        &mut self,
        input: &I,
        output: &mut O,
    ) where
        I: AsRef<[T]> + ?Sized,
        O: AsMut<[T]> + ?Sized,
    {
        let input_samples = input.as_ref().len() / self.channels as usize;
        let output_samples = output.as_mut().len() / self.channels as usize;

        self.process_internal(
            Some(input),
            input_samples,
            output,
            output_samples,
        );
    }

    pub fn process_ex<I, O>(
        &mut self,
        input: &I,
        input_samples: usize,
        outputs: &mut O,
        output_samples: usize,
    ) where
        I: AsRef<[T]> + ?Sized,
        O: AsMut<[T]> + ?Sized,
    {
        self.process_internal(
            Some(input),
            input_samples,
            outputs,
            output_samples,
        );
    }

    pub fn flush<O>(&mut self, output: &mut O, playback_rate: T)
    where
        O: AsMut<[T]> + ?Sized,
    {
        let output_samples = output.as_mut().len() / self.channels as usize;
        self.flush_ex(output, output_samples, playback_rate);
    }

    pub fn flush_ex<O>(
        &mut self,
        outputs: &mut O,
        output_samples: usize,
        playback_rate: T,
    ) where
        O: AsMut<[T]> + ?Sized,
    {
        let output_block = (output_samples - self.stft.default_intervals()).max(0);
        if output_block > 0 {
            let input = None::<&[T]>;
            let input_samples = (T::from_usize(output_block).unwrap() * playback_rate).to_usize().unwrap();

            self.process_internal(
                input,
                input_samples,
                outputs,
                output_block,
            );
        }

        let tail_samples = output_samples - output_block;
        self.stft.finish_output(Some(T::one()), None);
        self.temp_process_buffer.resize(tail_samples, T::zero());

        let output = outputs.as_mut();

        let mut temp_buffer = vec![T::zero(); tail_samples];
        for c in 0..self.channels as usize {
            let channel_index = c;

            self.stft
                .read_output(channel_index, tail_samples, &mut temp_buffer);

            for i in 0..tail_samples {
                output[(output_block + i) * self.channels as usize + channel_index] = temp_buffer[i];
            }

            self.stft
                .read_output_ex(channel_index, tail_samples, tail_samples, &mut temp_buffer);

            for i in 0..tail_samples {
                let value = temp_buffer[tail_samples - 1 - i];
                let output_index = (output_block + tail_samples - 1 - i) * self.channels as usize + channel_index;
                output[output_index] = output[output_index] - value;
            }
        }

        self.stft.reset(T::from_f32(0.01));
        self.channel_bands.reset();
    }

    pub fn exact<I, O>(
        &mut self,
        input: &I,
        output: &mut O,
    ) where
        I: AsRef<[T]> + ?Sized,
        O: AsMut<[T]> + ?Sized,
    {
        let input_samples = input.as_ref().len() / self.channels as usize;
        let output_samples = output.as_mut().len() / self.channels as usize;
        self.exact_ex(input, input_samples, output, output_samples);
    }

    pub fn exact_ex<I, O>(
        &mut self,
        inputs: &I,
        input_samples: usize,
        outputs: &mut O,
        output_samples: usize,
    ) -> bool
    where
        I: AsRef<[T]> + ?Sized,
        O: AsMut<[T]> + ?Sized,
    {
        let ratio = T::from_f32(output_samples as f32 / input_samples as f32).unwrap();
        let seek_length = self.output_seek_length(ratio);

        if input_samples < seek_length as usize {
            return false;
        }

        self.output_seek(&inputs, seek_length as usize);

        let output_index = output_samples - (T::from_i32(seek_length).unwrap() / ratio).to_usize().unwrap();

        let input_slice_len = input_samples - seek_length as usize;
        let inputs = inputs.as_ref();
        self.process_internal(
            Some(&inputs[seek_length as usize..]),
            input_slice_len,
            outputs,
            output_index,
        );

        let flush_length = output_samples - output_index;
        let outputs = outputs.as_mut();

        self.flush_ex(
            &mut outputs[output_index * self.channels as usize..],
            flush_length,
            ratio,
        );

        true
    }

    pub(crate) fn band_to_freq(&self, band: T) -> T {
        self.stft.bin_to_freq(band)
    }

    pub(crate) fn freq_to_band(&self, freq: T) -> T {
        self.stft.freq_to_bin(freq)
    }

    pub(crate) fn update_process_spectrum_steps(&mut self) {
        let process_spectrum_steps = &mut self.process_spectrum_steps;
        let block_process = &self.block_process;

        *process_spectrum_steps = 0;
        if block_process.new_spectrum {
            *process_spectrum_steps += self.channels as usize;
        }

        if block_process.mapped_frequencies {
            *process_spectrum_steps += SMOOTH_ENERY_STEPS;
            *process_spectrum_steps += 1; // find peaks
        }

        *process_spectrum_steps += 1; // Updating the output map
        *process_spectrum_steps += self.channels as usize; // Preliminary Phase-Vocoder prediction
        *process_spectrum_steps += SPLIT_MAIN_PREDICTION;

        if block_process.new_spectrum {
            *process_spectrum_steps += 1; // Input -> PrevInput
        }

        if block_process.process_formanets {
            *process_spectrum_steps += 3;
        }
    }

    pub(crate) fn process_spectrum(&mut self, mut step: usize) {
        let mut time_factor = self.block_process.time_factor;

        let smoothing_bins = T::from_f32(self.stft.fft_samples() as f32 
            / self.stft.default_intervals() as f32).unwrap();
            
        let long_vertical_step = smoothing_bins.round();

        let max_clean_stretch = T::from_f32(MAX_CLEAN_STRETCH).unwrap();

        time_factor = time_factor.max(T::one() / max_clean_stretch);
        let random_time_factor = time_factor > max_clean_stretch;

        let uniform_start = if random_time_factor {
            time_factor - max_clean_stretch * T::from_f32(2.0).unwrap()
        } else {
            T::zero()
        };

        let uniform_end = time_factor;

        let uniform_distributon = rand::distr::Uniform::new(uniform_start, uniform_end).unwrap();

        let mut max_output = Complex::zero();

        if self.block_process.new_spectrum {
            if step < self.channels as usize {
                let channel = step;

                let mut rot = Complex::from_polar(T::one(), {
                    self.band_to_freq(T::zero())
                        * T::from_usize(self.stft.default_intervals()).unwrap()
                        * T::from_f32(2.0 * std::f32::consts::PI).unwrap()
                });
                let freq_step = self.band_to_freq(T::one()) - self.band_to_freq(T::zero());
                let rot_step = Complex::from_polar(T::one(), {
                    freq_step * T::from_f32(self.stft.default_intervals() as f32 * (2.0 * std::f32::consts::PI)).unwrap()
                });

                let bins = self.channel_bands.bands_for_channel_mut(channel);
                for b in 0..self.bands as usize {
                    let band = &mut bins[b];

                    band.output = misc::mul(band.output, rot, false);
                    band.prev_input = misc::mul(band.prev_input, rot, false);

                    rot = misc::mul(rot, rot_step, false);
                }

                return;
            }

            step -= self.channels as usize;
        }

        if self.block_process.mapped_frequencies {
            if step < SMOOTH_ENERY_STEPS {
                self.smooth_energy(step, smoothing_bins);
                return;
            }

            step -= SMOOTH_ENERY_STEPS;

            if misc::post_decrement(&mut step) == 0 {
                self.find_peaks();
                return;
            }
        }

        if misc::post_decrement(&mut step) == 0 {
            if self.block_process.mapped_frequencies {
                self.update_output_map();
            } else {
                for b in 0..self.bands as usize {
                    self.output_map[b] = PitchMapPoint {
                        input_bin: T::from_usize(b).unwrap(),
                        freq_grad: T::one(),
                    };
                }

                for c in 0..self.channels as usize {
                    let binds = self.channel_bands.bands_for_channel_mut(c);

                    for b in 0..self.bands as usize {
                        let band = &mut binds[b];
                        band.input_energy = band.input.norm_sqr();
                    }
                }
            }

            return;
        }

        if self.block_process.process_formanets {
            if step < 3 {
                self.update_formants(step);
                return;
            }

            step -= 3;
        }

        if step < self.channels as usize {
            let c = step;
            let noise_floor = T::from_f32(NOISE_FLOOR).unwrap();

            for b in 0..self.bands as usize {
                let map_point = self.output_map[b];
                let low_index = map_point.input_bin.to_usize().unwrap_or(0);
                let frac_index = map_point.input_bin - T::from_usize(low_index).unwrap();

                let predictions = &mut self.channel_predictions.predictions_for_channel(c);

                let prediction = &mut predictions[b];
                let prev_energy = prediction.energy;
                prediction.energy =
                    self.channel_bands
                        .get_fractional_sample_ex(c, low_index, frac_index, |band| {
                            band.input_energy
                        });

                prediction.energy = (prediction.energy * map_point.freq_grad).max(T::zero());
                prediction.input =
                    self.channel_bands
                        .get_fractional_ex(c, low_index, frac_index, |band| band.input);

                let prev_input =
                    self.channel_bands
                        .get_fractional_ex(c, low_index, frac_index, |band| band.prev_input);

                let bins = self.channel_bands.bands_for_channel_mut(c);
                let band = &mut bins[b];

                let freq_twist = misc::mul(prediction.input, prev_input, true);

                let phase = misc::mul(band.output, freq_twist, false);
                band.output = phase / ((prev_energy.max(prediction.energy)) + noise_floor);

                max_output = misc::max_mag(max_output, band.output);
            }

            return;
        }

        step -= self.channels as usize;

        if step < SPLIT_MAIN_PREDICTION {
            let chunk = step;

            let startb = (self.bands as usize * chunk) / SPLIT_MAIN_PREDICTION;
            let endb = (self.bands as usize * (chunk + 1)) / SPLIT_MAIN_PREDICTION;

            for b in startb..endb {
                let bvalue = T::from_usize(b).unwrap();

                let mut max_energy = self.channel_predictions.predictions_for_channel(0)[b].energy;
                let mut max_channel = 0;

                for c in 1..self.channels as usize {
                    let energy = self.channel_predictions.predictions_for_channel(c)[b].energy;
                    if energy > max_energy {
                        max_energy = energy;
                        max_channel = c;
                    }
                }

                let predictions = self
                    .channel_predictions
                    .predictions_for_channel(max_channel);
                let prediction = predictions[b];

                let mut phase = Complex::<T>::zero();
                let map_point = self.output_map[b];

                if bvalue > T::zero() {
                    let bin_time_factor = if random_time_factor {
                        self.random_engine.get_dist(&uniform_distributon)
                    } else {
                        time_factor
                    };

                    let down_input = self.channel_bands.get_fractional(
                        max_channel,
                        map_point.input_bin - bin_time_factor,
                        |band| band.input,
                    );

                    let short_vertical_twist = misc::mul(prediction.input, down_input, true);

                    phase = phase + {
                        let down_bin = self.channel_bands.bands_for_channel(max_channel);
                        let down_band = &down_bin[b - 1];

                        misc::mul(down_band.output, short_vertical_twist, false)
                    };

                    // Upwards vertical steps
                    if bvalue as T >= long_vertical_step {
                        let long_down_input = self.channel_bands.get_fractional(
                            max_channel,
                            map_point.input_bin - long_vertical_step * bin_time_factor,
                            |band| band.input,
                        );

                        let long_vertical_twist =
                            misc::mul(prediction.input, long_down_input, true);

                        phase = phase + {
                            let down_bin = self.channel_bands.bands_for_channel(max_channel);
                            let down_band = &down_bin[b - long_vertical_step.to_usize().unwrap()];

                            misc::mul(down_band.output, long_vertical_twist, false)
                        };
                    }
                }

                // Downwards vertical steps
                let bands = T::from_i32(self.bands - 1).unwrap();
                if bvalue < bands {
                    let up_prediction = predictions[b + 1];
                    let up_map_point = self.output_map[b + 1];

                    let bin_time_factor = if random_time_factor {
                        self.random_engine.get_dist(&uniform_distributon)
                    } else {
                        time_factor
                    };

                    let down_input = self.channel_bands.get_fractional(
                        max_channel,
                        up_map_point.input_bin - bin_time_factor,
                        |band| band.input,
                    );

                    let short_vertical_twist = misc::mul(up_prediction.input, down_input, true);

                    phase = phase + {
                        let up_bin = self.channel_bands.bands_for_channel(max_channel);
                        let up_band = &up_bin[b + 1];

                        misc::mul(up_band.output, short_vertical_twist, true)
                    };

                    let bands = T::from_i32(self.bands).unwrap() - long_vertical_step;
                    if (bvalue as T) < bands {
                        let long_vertical_step_idx = long_vertical_step.to_usize().unwrap();
                        let long_up_prediction = predictions[b + long_vertical_step_idx];
                        let long_up_map_point = self.output_map[b + long_vertical_step_idx];

                        let long_down_input = self.channel_bands.get_fractional(
                            max_channel,
                            long_up_map_point.input_bin - long_vertical_step * bin_time_factor,
                            |band| band.input,
                        );

                        let long_vertical_twist =
                            misc::mul(long_up_prediction.input, long_down_input, true);

                        phase = phase + {
                            // dbg!(b, bands, long_vertical_step_idx);

                            let up_bin = self.channel_bands.bands_for_channel(max_channel);
                            let up_band = &up_bin[b + long_vertical_step_idx];

                            misc::mul(up_band.output, long_vertical_twist, true)
                        };
                    }
                }

                let output_bin = {
                    let bins = self.channel_bands.bands_for_channel_mut(max_channel);
                    let band = &mut bins[b];

                    band.output = prediction.make_output(phase);
                    max_output = misc::max_mag(max_output, band.output);

                    band.clone()
                };

                for c in 0..self.channels as usize {
                    if c != max_channel {
                        let channel_prediction =
                            &self.channel_predictions.predictions_for_channel(c)[b];
                        let channel_bin = &mut self.channel_bands.bands_for_channel_mut(c)[b];

                        let channel_twist =
                            misc::mul(channel_prediction.input, prediction.input, true);

                        let channel_phase = misc::mul(output_bin.output, channel_twist, false);

                        channel_bin.output = channel_prediction.make_output(channel_phase);

                        max_output = misc::max_mag(max_output, channel_bin.output);
                    }
                }
            }

            return;
        }

        step -= SPLIT_MAIN_PREDICTION;

        if self.block_process.new_spectrum {
            if misc::post_decrement(&mut step) == 0 {
                for bin in self.channel_bands.channel_bands.iter_mut() {
                    bin.prev_input = bin.input;
                }
            }
        }

        // println!("Max output mag: {}", max_output.norm_sqr());
    }

    pub(crate) fn smooth_energy(&mut self, mut step: usize, smoothing_bins: T) {
        let smoothing_slew = T::from_f32(1.0 / (1.0 + smoothing_bins.to_f32().unwrap() * 0.5)).unwrap();

        if misc::post_decrement(&mut step) == 0 {
            self.energy.fill(T::zero());

            for c in 0..self.channels as usize {
                let binds = self.channel_bands.bands_for_channel_mut(c);

                for b in 0..self.bands as usize {
                    let band = &mut binds[b];

                    let e = band.input.norm_sqr();
                    band.input_energy = e;
                    self.energy[b] = self.energy[b] + e;
                }
            }

            for b in 0..self.bands as usize {
                self.smoothed_energy[b] = self.energy[b];
            }

            self.smooth_energy_state = T::zero();
            return;
        }

        let mut smoothed = self.smooth_energy_state;

        // Backward smoothing
        for b in (0..self.bands as usize).rev() {
            smoothed = smoothed + ((self.smoothed_energy[b] - smoothed) * smoothing_slew);
            self.smoothed_energy[b] = smoothed;
        }

        // Forward smoothing
        for b in 0..self.bands as usize {
            smoothed = smoothed + ((self.smoothed_energy[b] - smoothed) * smoothing_slew);
            self.smoothed_energy[b] = smoothed;
        }

        self.smooth_energy_state = smoothed;
    }

    pub(crate) fn map_freq(&self, freq: T) -> T {
        if let Some(custom_map) = &self.custom_freq_map {
            return custom_map(freq);
        }

        if freq > self.freq_tonality_limit {
            return freq + (self.freq_multiplier - T::one()) * self.freq_tonality_limit;
        }

        freq * self.freq_multiplier
    }

    pub(crate) fn find_peaks(&mut self) {
        self.peaks.resize(0, Peak::default());

        let mut start = 0;
        while start < self.bands as usize {
            if self.energy[start] > self.smoothed_energy[start] {
                let mut end = start;

                let mut band_sum = T::zero();
                let mut energy_sum = T::zero();

                while end < self.bands as usize && self.energy[end] > self.smoothed_energy[end] {
                    let endt = T::from_usize(end).unwrap();
                    
                    band_sum = band_sum + (endt * self.energy[end]);
                    energy_sum = energy_sum + self.energy[end];
                    end += 1;
                }

                let avg_band = band_sum / energy_sum;
                let avg_freq = self.band_to_freq(avg_band);
                self.peaks.push(Peak {
                    input: avg_band,
                    output: self.freq_to_band(self.map_freq(avg_freq)),
                });

                start = end;
            } else {
                start += 1;
            }
        }
    }

    pub(crate) fn update_output_map(&mut self) {
        if self.peaks.is_empty() {
            for b in 0..self.bands as usize {
                self.output_map[b] = PitchMapPoint {
                    input_bin: T::from_usize(b).unwrap(),
                    freq_grad: T::one(),
                };
            }
            return;
        }

        let bottom_offset = self.peaks[0].input - self.peaks[0].output;
        let max_band_iter = self.bands.min(
            self.peaks[0]
            .output.ceil()
            .max(T::zero())
            .to_i32()
            .unwrap()
        ) as usize;

        for b in 0..max_band_iter {
            self.output_map[b] = PitchMapPoint {
                input_bin: T::from_usize(b).unwrap() + bottom_offset,
                freq_grad: T::one(),
            };
        }

        for p in 1..self.peaks.len() {
            let prev = &self.peaks[p - 1];
            let next = &self.peaks[p];

            let range_scale = T::one() / (next.output - prev.output);
            let out_offset = prev.input - prev.output;
            let out_scale = (next.input - next.output) - (prev.input - prev.output);
            let grad_scale = out_scale * range_scale;

            let start_bin = T::zero().max(prev.output.ceil()).to_i32().unwrap() as usize;
            let end_bin = self.bands.min(next.output.ceil().to_i32().unwrap()) as usize;

            let six = T::from_f32(6.0).unwrap();
            let three = T::from_f32(3.0).unwrap();
            let two = T::from_f32(2.0).unwrap();

            for b in start_bin..end_bin {
                let bvalue = T::from_usize(b).unwrap();

                let r = (bvalue - prev.output) * range_scale;
                let h = r * r * (three - two * r); // Hermite interpolation
                let out_b = bvalue + out_offset + out_scale * h;

                let grad_h = six * r * (T::one() - r);
                let grad_b = T::one() + grad_scale * grad_h;

                self.output_map[b] = PitchMapPoint {
                    input_bin: out_b,
                    freq_grad: grad_b,
                };
            }
        }

        let back_peaks = &self.peaks[self.peaks.len() - 1];
        let top_offset = back_peaks.input - back_peaks.output;

        let start_top = (back_peaks.output.to_i32().unwrap() + 1).max(0) as usize;
    
        for b in start_top..self.bands as usize {
            let bvalue = T::from_usize(b).unwrap();

            self.output_map[b as usize] = PitchMapPoint {
                input_bin: bvalue + top_offset,
                freq_grad: T::one(),
            };
        }
    }

    pub(crate) fn inv_map_formant(&self, freq: T) -> T {
        if freq * self.inv_formant_multiplier > self.freq_tonality_limit {
            return freq + (T::one() - self.formant_multiplier) * self.freq_tonality_limit;
        }

        freq * self.inv_formant_multiplier
    }

    pub(crate) fn estimate_frequency(&mut self) -> T {
        let mut peak_indices: [usize; 3] = [0, 0, 0];
        for b in 1..self.bands as usize - 1 {
            let e = self.formant_metrics[b];

            // Local maxima only
            if e < self.formant_metrics[b - 1] || e < self.formant_metrics[b + 1] {
                continue;
            }

            if e > self.formant_metrics[peak_indices[0]] {
                if e > self.formant_metrics[peak_indices[1]] {
                    if e > self.formant_metrics[peak_indices[2]] {
                        peak_indices = [peak_indices[1], peak_indices[2], b];
                    } else {
                        peak_indices = [peak_indices[1], b, peak_indices[2]];
                    }
                } else {
                    peak_indices[0] = b;
                }
            }
        }

        // Very rough pitch estimation
        let mut peak_estimate = peak_indices[2] as i32;
        let dotone = T::from_f32(0.1).unwrap();

        if self.formant_metrics[peak_indices[1]] > self.formant_metrics[peak_indices[2]] * dotone {
            let diff = (peak_estimate as i32 - peak_indices[1] as i32).abs();
            if diff > peak_estimate as i32 / 8 && diff < peak_estimate as i32 * 7 / 8 {
                peak_estimate = peak_estimate % diff;
            }

            if self.formant_metrics[peak_indices[0]] > self.formant_metrics[peak_indices[2]] * dotone {
                let diff = (peak_estimate as i32 - peak_indices[0] as i32).abs();
                if diff > peak_estimate as i32 / 8 && diff < peak_estimate as i32 * 7 / 8 {
                    peak_estimate = peak_estimate % diff;
                }
            }
        }

        let weight = self.formant_metrics[peak_indices[2]];

        let dottwohalf = T::from_f32(0.25).unwrap();
        let peak_estimate = T::from_i32(peak_estimate).unwrap();

        self.freq_estimate_weighted
            = self.freq_estimate_weighted + (peak_estimate * weight - self.freq_estimate_weighted) * dottwohalf;

        self.freq_estimate_weight = self.freq_estimate_weight + (weight - self.freq_estimate_weight) * dottwohalf;
        
        self.freq_estimate_weighted / (self.freq_estimate_weight + T::from_f32(1e-30).unwrap())
    }

    pub(crate) fn update_formants(&mut self, mut step: usize) {
        if misc::post_decrement(&mut step) == 0 {
            self.formant_metrics.fill(T::zero());

            for c in 0..self.channels as usize {
                let binds = self.channel_bands.bands_for_channel(c);
                for b in 0..self.bands as usize {
                    self.formant_metrics[b] = self.formant_metrics[b] + binds[b].input_energy;
                }
            }

            self.freq_estimate = self.freq_to_band(self.formant_base_freq);
            if self.formant_base_freq <= T::zero() {
                self.freq_estimate = self.estimate_frequency();
            }
        } else if misc::post_decrement(&mut step) == 0 {
            let mut decay = T::from_f32(1.0 - 1.0 / (self.freq_estimate.to_f32().unwrap() * 0.5 + 1.0)).unwrap();
            let mut e = T::zero();

            for _ in 0..2 {
                for b in (0..self.bands as usize).rev() {
                    e = self.formant_metrics[b].max(e * decay);
                    self.formant_metrics[b] = e;
                }
                for b in 0..self.bands as usize {
                    e = self.formant_metrics[b].max(e * decay);
                    self.formant_metrics[b] = e;
                }
            }

            decay = T::one() / decay;

            for _ in 0..2 {
                for b in (0..self.bands as usize).rev() {
                    e = self.formant_metrics[b].min(e * decay);
                    self.formant_metrics[b] = e;
                }
                for b in 0..self.bands as usize {
                    e = self.formant_metrics[b].min(e * decay);
                    self.formant_metrics[b] = e;
                }
            }
        } else {
            let bands = T::from_i32(self.bands).unwrap();
            let get_formant = |mut band: T| -> T {
                if band < T::zero() {
                    return T::zero();
                }

                band = band.min(bands);
                let band_floor = band.floor().to_usize().unwrap();
                let frac_band = band - band.floor();
                let low = self.formant_metrics[band_floor];
                let high = self.formant_metrics[band_floor + 1];

                low + (high - low) * frac_band
            };

            for b in 0..self.bands as usize {
                let bvalue = T::from_usize(b).unwrap();

                let input_f = self.band_to_freq(bvalue);
                let mut output_f = if self.formant_compensation {
                    self.map_freq(input_f)
                } else {
                    input_f
                };

                output_f = self.inv_map_formant(output_f);

                let input_e = self.formant_metrics[b];
                let target_e = get_formant(self.freq_to_band(output_f));

                let formant_ratio = target_e / (input_e + T::from_f32(1e-30).unwrap());
                let energy_ratio = formant_ratio;

                for c in 0..self.channels as usize {
                    let binds = self.channel_bands.bands_for_channel_mut(c);
                    binds[b].input_energy = binds[b].input_energy * energy_ratio;
                }
            }
        }
    }
}

const SPLIT_MAIN_PREDICTION: usize = 8;
const SMOOTH_ENERY_STEPS: usize = 3;

#[derive(Debug, Clone)]
pub(crate) struct BlockProcess<T: Sample> {
    pub samples_since_last: usize,
    pub steps: usize,
    pub step: usize,

    pub new_spectrum: bool,
    pub reanalyze_prev: bool,
    pub mapped_frequencies: bool,
    pub process_formanets: bool,
    pub time_factor: T,
}

impl<T: Sample> Default for BlockProcess<T> {
    fn default() -> Self {
        BlockProcess {
            samples_since_last: usize::MAX,
            steps: 0,
            step: 0,
            new_spectrum: false,
            reanalyze_prev: false,
            mapped_frequencies: false,
            process_formanets: false,
            time_factor: T::zero(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Peak<T: Sample> {
    pub input: T,
    pub output: T,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PitchMapPoint<T: Sample> {
    pub input_bin: T,
    pub freq_grad: T,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Band<T: Sample> {
    pub input: Complex<T>,
    pub prev_input: Complex<T>,
    pub output: Complex<T>,
    pub input_energy: T,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Prediction<T: Sample> {
    pub energy: T,
    pub input: Complex<T>,
}

impl<T: Sample> Prediction<T> {
    pub fn make_output(&self, mut phase: Complex<T>) -> Complex<T> {
        let mut phase_norm = phase.norm_sqr();
        let noise_floor = T::from_f32(NOISE_FLOOR).unwrap();

        if phase_norm <= noise_floor {
            phase = self.input;
            phase_norm = phase.norm_sqr() + noise_floor;
        }

        let value = phase * (self.energy / phase_norm).sqrt();
        value
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StretchBands<T: Sample> {
    pub channel_bands: Vec<Band<T>>,
    pub bands: usize,
}

#[allow(dead_code)]
impl<T: Sample> StretchBands<T> {
    pub fn new(channels: usize, bands: usize) -> Self {
        StretchBands {
            channel_bands: vec![Band::default(); channels * bands],
            bands,
        }
    }

    pub fn resize(&mut self, channels: usize, bands: usize) {
        self.channel_bands.resize(channels * bands, Band::default());
        self.bands = bands;
    }

    pub fn reset(&mut self) {
        for b in self.channel_bands.iter_mut() {
            *b = Band::default();
        }
    }

    pub fn bands_for_channel(&self, channel: usize) -> &[Band<T>] {
        let start = channel * self.bands as usize;
        let end = start + self.bands as usize;
        &self.channel_bands[start..end]
    }

    pub fn bands_for_channel_mut(&mut self, channel: usize) -> &mut [Band<T>] {
        let start = channel * self.bands as usize;
        let end = start + self.bands as usize;
        &mut self.channel_bands[start..end]
    }

    pub fn get_band<F>(&self, channel: usize, index: usize, band: F) -> Complex<T>
    where
        F: Fn(&Band<T>) -> Complex<T>,
    {
        if index > isize::MAX as usize || index >= self.bands as usize {
            return Complex::zero();
        }

        let band_index = band(&self.channel_bands[(channel * self.bands as usize) + index]);

        band_index
    }

    pub fn get_fractional<F>(&self, channel: usize, input_index: T, band: F) -> Complex<T>
    where
        F: Fn(&Band<T>) -> Complex<T>,
    {
        let low_index = input_index.to_f32().unwrap().floor() as usize;
        let frac = input_index - T::from_usize(low_index).unwrap();

        self.get_fractional_ex(channel, low_index, frac, band)
    }

    pub fn get_fractional_ex<F>(
        &self,
        channel: usize,
        low_index: usize,
        frac: T,
        band: F,
    ) -> Complex<T>
    where
        F: Fn(&Band<T>) -> Complex<T>,
    {
        if low_index > isize::MAX as usize || low_index >= self.bands as usize {
            return Complex::zero();
        }

        let low_band = self.get_band(channel, low_index, &band);
        let high_band = self.get_band(channel, low_index + 1, &band);

        low_band + (high_band - low_band) * frac
    }

    pub fn get_band_sample<F>(&self, channel: usize, index: usize, band: F) -> T
    where
        F: Fn(&Band<T>) -> T,
    {
        if index > isize::MAX as usize || index >= self.bands as usize {
            return T::zero();
        }

        let band_index = band(&self.channel_bands[(channel * self.bands as usize) + index]);

        band_index
    }

    #[inline(never)]
    pub fn get_fractional_sample<F>(&self, channel: usize, input_index: T, band: F) -> T
    where
        F: Fn(&Band<T>) -> T,
    {
        let low_index = input_index.to_f32().unwrap().floor() as usize;
        let frac = input_index - T::from_usize(low_index).unwrap();

        self.get_fractional_sample_ex(channel, low_index, frac, band)
    }

    #[inline(never)]
    pub fn get_fractional_sample_ex<F>(
        &self,
        channel: usize,
        low_index: usize,
        frac: T,
        band: F,
    ) -> T
    where
        F: Fn(&Band<T>) -> T,
    {
        let low_band = self.get_band_sample(channel, low_index, &band);
        let high_band = self.get_band_sample(channel, low_index + 1, &band);

        low_band + (high_band - low_band) * frac
    }
}

#[derive(Debug, Clone)]
pub(crate) struct StretchPredictions<T: Sample> {
    pub channel_predictions: Vec<Prediction<T>>,
    pub bands: usize,
}

#[allow(dead_code)]
impl<T: Sample> StretchPredictions<T> {
    pub fn new(channels: usize, bands: usize) -> Self {
        StretchPredictions {
            channel_predictions: vec![Prediction::default(); channels * bands],
            bands,
        }
    }

    pub fn resize(&mut self, channels: usize, bands: usize) {
        self.channel_predictions
            .resize(channels * bands, Prediction::default());
        self.bands = bands;
    }

    pub fn reset(&mut self) {
        for p in self.channel_predictions.iter_mut() {
            *p = Prediction::default();
        }
    }

    pub fn predictions_for_channel(&mut self, channel: usize) -> &mut [Prediction<T>] {
        &mut self.channel_predictions[(channel * self.bands as usize)..]
    }
}

pub(crate) enum RandomType {
    None,
    RngThread(rand::rngs::ThreadRng),
    Seedable(rand::rngs::StdRng),
}

impl std::fmt::Debug for RandomType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RandomType::None => write!(f, "None"),
            RandomType::RngThread(_) => write!(f, "RngThread"),
            RandomType::Seedable(_) => write!(f, "Seedable"),
        }
    }
}

impl Clone for RandomType {
    fn clone(&self) -> Self {
        match self {
            RandomType::None => RandomType::None,
            RandomType::RngThread(_) => RandomType::RngThread(rand::rngs::ThreadRng::default()),
            RandomType::Seedable(_) => RandomType::Seedable(rand::rngs::StdRng::seed_from_u64(0)),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RandomEngine {
    pub rng: RandomType,
}

impl RandomEngine {
    pub fn from_seed(seed: u64) -> Self {
        RandomEngine {
            rng: RandomType::Seedable(rand::rngs::StdRng::seed_from_u64(seed)),
        }
    }

    pub fn new() -> Self {
        RandomEngine {
            rng: RandomType::RngThread(rand::rngs::ThreadRng::default()),
        }
    }

    pub fn get_dist<F, T>(&mut self, distr: &F) -> T
    where
        F: rand::distr::Distribution<T>,
    {
        match &mut self.rng {
            RandomType::RngThread(rng) => distr.sample(rng),
            RandomType::Seedable(rng) => distr.sample(rng),
            RandomType::None => panic!("No RNG available"),
        }
    }
}

impl Default for RandomEngine {
    fn default() -> Self {
        RandomEngine {
            rng: RandomType::None,
        }
    }
}