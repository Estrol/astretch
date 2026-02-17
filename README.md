# AStretch (Audio Stretch)
Rust clone/port of [Signalsmith Stretch](https://github.com/signalsmith/signalsmith-stretch) library, which does time stretching and pitch shifting using phase-vocoder on audio signals.

This library uses [rustfft](https://github.com/ejmahler/RustFFT) for faster FFT operations.

## Supported types:
- `f32`
- `f64`
- or any floating point type that implements `astretch::dsp::Sample` trait.

## Configuring:
The easiest way to configure is a .preset???() method:
```rust
let mut stretch = Stretch::new();
stretch.preset_default(2, 44100.0, split_compute);
```
or use custom configuration (or optionally use custom random seeding):
```rust
let mut stretch = Stretch::from_seed(12345);
stretch.configure(
    2, // num channels
    block_samples, // block size in samples
    interval_samples, // interval size in samples
    split_compute, // allow faster computation at the cost of more latency
)
```

## Pitch shifting:
The easiest way to pitch shift is to set the stretch factor:
```rust
stretch.set_transpose_factor(2.0, None); // one octave up
// or
stretch.set_transpose_factor(4.0, Some(8000 / sample_rate)); // Uses "tonality limit"
```
or use callbacks:
```rust
stretch.set_freq_map(|input_freq| {
    // input_freq is the frequency of the current band in the input signal
    // return the desired frequency for that band in the output signal
    input_freq * 2.0 // one octave up
});
```

## Time stretching:
The easiest way to time stretch is to set the stretch factor:
```rust
let stretch_factor = 1.5;
let data = [...];

// Less than input frames: speed up, more than input frames: slow down
let mut output = vec![0.0; (data.len() as f32 * stretch_factor) as usize];
stretch.process(&data, &mut output);
```

## Latency
You can get the latency by using these methods:
```rust
let input = stretch.input_latency();
let output = stretch.output_latency();
```

If you processing fixed number of frames, you can use the `exact` method instead of `process` see [below](#fixed-time-processing).

### Split computation
All the configuration `.preset???()` and `.configure()` methods have an optional `split_compute` parameter. When enabled, this introduces more latency at exactly one interval of output latency and uses this to spread the computation out more evenly, which can reduce the CPU load at the cost of more latency. This is disabled by default, but can be enabled like this:
```rust
stretch.preset_default(2, 44100.0, true);
```
or
```rust
stretch.configure(
    2, // num channels
    block_samples, // block size in samples
    interval_samples, // interval size in samples
    true, // split compute
)
```

# Fixed time processing:
To process a fixed number of frames, use the `exact` method instead:
```rust
let stretch_factor = 1.5;
let data = [...];
let mut output = vec![0.0; (data.len() as f32 / stretch_factor) as usize];

stretch.set_transpose_factor(stretch_factor);
stretch.exact(&data, &mut output);
```

## Performance
The library use `rustfft` for faster FFT operations, while the `rustfft` might use SIMD operations, the other parts of the library (like `DynamicSTFT` or `RealFFT`) are not optimized for SIMD (or still in scalar mode), so the performance might not be as good as the other libraries that are optimized for SIMD, but it should still be faster.

## License
MIT License

## Acknowledgements
- [Signalsmith Stretch](https://github.com/signalsmith/signalsmith-stretch) - The original library that this is based on, also
whos helped me diagnose and fix bugs in this implementation.
- [rustfft](https://github.com/ejmahler/RustFFT) - The FFT library used for faster FFT operations on supported machines.