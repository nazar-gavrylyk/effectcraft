//! H.264 on the machine's video hardware, through a hook the app implements (the export crate
//! builds for the web and starts no processes itself).

use craft_media::isobmff::{AvcConfig, SampleEntry};
use effectcraft_project::render_queue::OutputFormat;
use effectcraft_time::FrameRate;

use crate::encode::{Codes, Packet, VideoEncoder, Yuv420, keyint, tag_nclx_limited};
use crate::{Cx, ExportError, Result};

/// What the hardware encoder is asked for: 8-bit 4:2:0 H.264 High, limited range, with the
/// colour code points of [`crate::encode::output_codes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HardwareRequest {
    pub width: u32,
    pub height: u32,
    /// Frame rate as numerator and denominator; packet times are in 1 / `rate[0]` seconds.
    pub rate: [u32; 2],
    pub bitrate_kbps: u32,
    /// Frames between key frames.
    pub keyint: u32,
    pub primaries: u8,
    pub transfer: u8,
    pub matrix: u8,
}

/// One encoded frame. `pts` and `dts` are in 1 / `rate[0]` seconds; `dts` is negative while the
/// encoder holds frames back. The first packet's `pts - dts` is [`HardwareJob::delay`] frames
/// (`delay * rate[1]`): the export starts the movie that far in, so another offset is an error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HardwarePacket {
    /// Length-prefixed NAL units, one MP4 sample.
    pub data: Vec<u8>,
    pub pts: i64,
    pub dts: i64,
    pub key: bool,
}

/// The answer to [`HardwareVideo::open`].
pub enum HardwareOffer {
    Job(Box<dyn HardwareJob>),
    /// Hardware cannot take this export (no helper installed, a size the device cannot do): the
    /// built-in encoder runs and the render log says why.
    Declined(String),
}

/// Starts hardware encodes. A refusal that is not a decline (the helper broke) is an `Err`, which
/// fails the export: it never switches to software silently.
pub trait HardwareVideo: Send + Sync {
    fn open(&self, request: &HardwareRequest) -> Result<HardwareOffer>;
}

/// One open hardware encode. The export calls [`HardwareJob::cancel`] on a job it abandons.
pub trait HardwareJob: Send {
    /// What encodes, for the render log ("VA-API (Intel Iris Xe Graphics), helper 0.1.0").
    fn description(&self) -> String;
    /// The `avcC` record.
    fn codec_config(&self) -> Vec<u8>;
    /// Frames the encoder holds back (B-frame reordering); 0 for none. The first packet must
    /// have `pts - dts` equal to `delay * rate[1]`.
    fn delay(&self) -> u32;
    /// An 8-bit 4:2:0 picture, planes tightly packed.
    fn submit(&mut self, y: &[u8], u: &[u8], v: &[u8]) -> Result<()>;
    /// The packets that are ready.
    fn take_packets(&mut self) -> Result<Vec<HardwarePacket>>;
    /// End the stream and return the remaining packets.
    fn finish(&mut self) -> Result<Vec<HardwarePacket>>;
    /// Stop the job and free the device. Also called after a failed `finish`, so it must be
    /// harmless when the job has already ended or failed, and when called twice.
    fn cancel(&mut self);
}

/// Note why a hook is not used for a format other than H.264.
pub(crate) fn note_format(cx: &Cx) {
    let format = cx.output.format;
    if cx.options.hardware.is_some() && format != OutputFormat::H264 {
        cx.note_hardware(Err(format!("hardware encoding is for H.264 only, not {}", format.label())));
    }
}

/// The hardware encoder for an H.264 export, or `None` for software (Off, or declined and
/// noted in the log).
pub(crate) fn open(cx: &Cx, w: u32, h: u32, rate: FrameRate, codes: Codes) -> Result<Option<HardwareH264>> {
    let Some(hook) = cx.options.hardware else { return Ok(None) };
    let (Ok(num), Ok(den)) = (u32::try_from(rate.num), u32::try_from(rate.den)) else {
        return Err(ExportError::Unsupported("the frame rate is too large for hardware encoding".into()));
    };
    let request = HardwareRequest {
        width: w,
        height: h,
        rate: [num, den],
        bitrate_kbps: cx.output.bitrate_kbps.max(100),
        keyint: keyint(rate),
        primaries: codes.primaries,
        transfer: codes.transfer,
        matrix: codes.matrix,
    };
    match hook.open(&request)? {
        HardwareOffer::Declined(why) => {
            cx.note_hardware(Err(why));
            Ok(None)
        }
        HardwareOffer::Job(job) => {
            cx.note_hardware(Ok(job.description()));
            HardwareH264::new(job, w, h, rate, codes).map(Some)
        }
    }
}

pub(crate) struct HardwareH264 {
    job: Box<dyn HardwareJob>,
    config: AvcConfig,
    w: u32,
    h: u32,
    rate: FrameRate,
    codes: Codes,
    finished: bool,
    /// The first packet has been checked against [`HardwareJob::delay`].
    checked: bool,
}

impl HardwareH264 {
    fn new(mut job: Box<dyn HardwareJob>, w: u32, h: u32, rate: FrameRate, codes: Codes) -> Result<HardwareH264> {
        let config = match AvcConfig::parse(&job.codec_config()) {
            Ok(config) => config,
            Err(e) => {
                job.cancel();
                return Err(ExportError::Encode(format!("the hardware encoder's codec configuration is invalid: {e}")));
            }
        };
        Ok(HardwareH264 { job, config, w, h, rate, codes, finished: false, checked: false })
    }

    fn packets(&mut self, ps: Vec<HardwarePacket>) -> Result<Vec<Packet>> {
        let expected = i64::from(self.job.delay()) * self.rate.den;
        let mut out = Vec::with_capacity(ps.len());
        for p in ps {
            let cto = p.pts.checked_sub(p.dts).and_then(|d| i32::try_from(d).ok());
            let cto = cto.ok_or_else(|| ExportError::Encode(format!("the hardware encoder gave a frame time out of range (pts {}, dts {})", p.pts, p.dts)))?;
            if !self.checked && i64::from(cto) != expected {
                return Err(ExportError::Encode(format!(
                    "the hardware encoder reported a delay of {} frames, but its first frame has pts {} and dts {}; pts - dts must be {expected}",
                    self.job.delay(),
                    p.pts,
                    p.dts
                )));
            }
            self.checked = true;
            out.push(Packet { data: p.data, key: p.key, cto });
        }
        Ok(out)
    }
}

impl VideoEncoder for HardwareH264 {
    fn sample_entry(&self) -> SampleEntry {
        let mut entry = SampleEntry::avc(self.config.clone(), self.w as u16, self.h as u16);
        tag_nclx_limited(&mut entry, self.codes);
        entry
    }
    fn encode(&mut self, _rgba: &[u8], _index: u64) -> Result<Vec<Packet>> {
        Err(ExportError::Unsupported("the hardware encoder takes 4:2:0 frames".into()))
    }
    fn takes_yuv420(&self) -> bool {
        true
    }
    fn encode_yuv420(&mut self, p: &Yuv420, _index: u64) -> Result<Vec<Packet>> {
        let (y, u, v) = p.slices(self.w as usize, self.h as usize);
        self.job.submit(y, u, v)?;
        let ps = self.job.take_packets()?;
        self.packets(ps)
    }
    fn flush(&mut self) -> Result<Vec<Packet>> {
        let ps = self.job.finish()?;
        self.finished = true;
        self.packets(ps)
    }
    fn media_start(&self) -> Option<i64> {
        (self.job.delay() > 0).then(|| i64::from(self.job.delay()) * self.rate.den)
    }
}

impl Drop for HardwareH264 {
    fn drop(&mut self) {
        if !self.finished {
            self.job.cancel();
        }
    }
}
