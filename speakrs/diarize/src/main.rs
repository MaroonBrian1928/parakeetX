//! Diarizes a 16 kHz mono PCM16 WAV with speakrs and prints RTTM to stdout.

use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use speakrs::{ExecutionMode, OwnedDiarizationPipeline};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let [_, mode, models_dir, wav] = args.as_slice() else {
        eprintln!("usage: speakrs-diarize <cpu|cuda> <models-dir> <audio.wav>");
        return ExitCode::from(2);
    };
    match run(mode, Path::new(models_dir), Path::new(wav)) {
        Ok(rttm) => {
            print!("{rttm}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("speakrs-diarize: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(mode: &str, models_dir: &Path, wav: &Path) -> Result<String> {
    let mode = match mode {
        "cpu" => ExecutionMode::Cpu,
        "cuda" => ExecutionMode::Cuda,
        other => return Err(format!("unknown mode {other:?}").into()),
    };
    let audio = read_pcm16_mono_16k(wav)?;
    let mut pipeline = OwnedDiarizationPipeline::from_dir(models_dir, mode)?;
    let rttm = pipeline.run(&audio)?.rttm("audio");
    // Tearing down the CUDA sessions segfaults with the sm_52 ONNX Runtime 1.20 build, and the
    // process exits right after printing, so leave cleanup to the OS.
    std::mem::forget(pipeline);
    Ok(rttm)
}

fn read_pcm16_mono_16k(path: &Path) -> Result<Vec<f32>> {
    let data = fs::read(path)?;
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }

    let mut format_ok = false;
    let mut pos = 12usize;
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into()?) as usize;
        let body = data.get(pos + 8..pos + 8 + size).ok_or("truncated WAV chunk")?;
        match id {
            b"fmt " => {
                let format = u16::from_le_bytes(body[0..2].try_into()?);
                let channels = u16::from_le_bytes(body[2..4].try_into()?);
                let rate = u32::from_le_bytes(body[4..8].try_into()?);
                let bits = u16::from_le_bytes(body[14..16].try_into()?);
                if format != 1 || channels != 1 || rate != 16_000 || bits != 16 {
                    return Err(format!(
                        "expected 16 kHz mono PCM16, got format={format} channels={channels} rate={rate} bits={bits}"
                    )
                    .into());
                }
                format_ok = true;
            }
            b"data" if format_ok => {
                return Ok(body
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&b| i16::from_le_bytes(b) as f32 / 32768.0)
                    .collect());
            }
            _ => {}
        }
        // RIFF chunks are word-aligned.
        pos += 8 + size + (size & 1);
    }
    Err("no fmt/data chunk found in WAV".into())
}
