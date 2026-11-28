//! Hardware H.264 through the `HardwareVideo` hook, with a stand-in that runs the built-in
//! encoder behind the hook's interface, so the muxing, the frame times, the `.part` file and the
//! fallbacks are checked without a helper process or a GPU.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use effectcraft_color::Label;
use effectcraft_export::render_queue::{OutputFormat, OutputModule, RenderSettings, TimeSpan};
use effectcraft_export::{ExportError, HardwareJob, HardwareOffer, HardwarePacket, HardwareRequest, HardwareVideo, Job, JobOptions, Report, export};
use effectcraft_media::MediaPool;
use effectcraft_project::{Comp, ItemId, ItemKind, LayerSource, Project, Solid, build};
use effectcraft_render::NoFootage;
use effectcraft_time::{FrameRate, Tick};

const W: u32 = 64;
const H: u32 = 48;
const BG: [f32; 3] = [0.1, 0.1, 0.4];
const RED_FROM: u64 = 5;

/// 64×48 at 10 fps, dark blue until frame 5, then a red bar through the centre.
fn project(frames: u64) -> (Project, ItemId) {
    let mut p = Project::default();
    let mut comp = Comp::new(W, H, FrameRate::new(10, 1), Tick::from_seconds_f64(frames as f64 / 10.0));
    comp.background = BG;
    let sid = p.add_item("Red", Label::Red, None, ItemKind::Solid(Solid { color: [1.0, 0.0, 0.0], width: 32, height: H, pixel_aspect: 1.0 }));
    let mut l = build::layer(&mut p, &comp, "Red", LayerSource::Solid { item: sid }, (32, H), None);
    l.in_point = comp.frame_rate.tick_of(RED_FROM as i64);
    comp.layers.push(l);
    let cid = p.add_item("Comp 1", Label::Sandstone, None, ItemKind::Comp(comp.into()));
    (p, cid)
}

#[derive(Clone, Copy, PartialEq)]
enum Fail {
    Never,
    /// `submit` of this frame.
    Submit(u64),
}

#[derive(Default)]
struct Calls {
    opened: AtomicUsize,
    cancelled: AtomicUsize,
    finished: AtomicUsize,
    /// At `finish`, the movie was at `<target>.part` and not yet at the target.
    part_only_at_finish: AtomicBool,
}

struct Stand {
    /// Frames `dts` runs behind `pts` at the start, which is what the job reports as its delay.
    lead: u32,
    fail: Fail,
    declines: Option<String>,
    /// Added to the delay the job reports, so it differs from the `dts` lead it produces.
    delay_error: u32,
    target: PathBuf,
    calls: Arc<Calls>,
}

impl Stand {
    fn new(target: &Path) -> Stand {
        Stand { lead: 0, fail: Fail::Never, declines: None, delay_error: 0, target: target.to_owned(), calls: Arc::default() }
    }
}

struct StandJob {
    enc: craft_video::h264::Encoder,
    den: i64,
    lead: u32,
    delay_error: u32,
    frame: u64,
    ready: Vec<HardwarePacket>,
    fail: Fail,
    target: PathBuf,
    calls: Arc<Calls>,
}

impl HardwareVideo for Stand {
    fn open(&self, r: &HardwareRequest) -> Result<HardwareOffer, ExportError> {
        self.calls.opened.fetch_add(1, Ordering::SeqCst);
        if let Some(why) = &self.declines {
            return Ok(HardwareOffer::Declined(why.clone()));
        }
        let mut cfg = craft_video::h264::EncoderConfig::new(r.width, r.height, r.rate[0], r.rate[1]);
        cfg.format = craft_video::h264::PacketFormat::LengthPrefixed;
        cfg.aud = false;
        cfg.keyint = r.keyint;
        cfg.bframes = if self.lead > 0 { 2 } else { 0 };
        cfg.rate = craft_video::h264::RateControl::Vbr { target_kbps: r.bitrate_kbps, max_kbps: r.bitrate_kbps * 3 / 2 };
        let enc = craft_video::h264::Encoder::new(cfg).map_err(|e| ExportError::Encode(e.to_string()))?;
        let job = StandJob {
            enc,
            den: i64::from(r.rate[1]),
            lead: self.lead,
            delay_error: self.delay_error,
            frame: 0,
            ready: vec![],
            fail: self.fail,
            target: self.target.clone(),
            calls: self.calls.clone(),
        };
        Ok(HardwareOffer::Job(Box::new(job)))
    }
}

impl StandJob {
    /// The built-in encoder's `dts` runs one frame behind `pts` when it reorders; a longer lead
    /// moves `dts` further back, as a deeper hardware pipeline does.
    fn packets(&self, ps: Vec<craft_video::h264::Packet>) -> Vec<HardwarePacket> {
        let extra = i64::from(self.lead.saturating_sub(1)) * self.den;
        ps.into_iter().map(|p| HardwarePacket { data: p.data, pts: p.pts, dts: p.dts - extra, key: p.keyframe }).collect()
    }
}

impl HardwareJob for StandJob {
    fn description(&self) -> String {
        "Stand-in API (Stand-in GPU), helper 9.9".into()
    }
    fn codec_config(&self) -> Vec<u8> {
        self.enc.avcc()
    }
    fn delay(&self) -> u32 {
        self.lead + self.delay_error
    }
    fn submit(&mut self, y: &[u8], u: &[u8], v: &[u8]) -> Result<(), ExportError> {
        if self.fail == Fail::Submit(self.frame) {
            return Err(ExportError::Encode("the video helper stopped".into()));
        }
        let frame = craft_video::h264::YuvFrame { y, u, v, y_stride: W as usize, uv_stride: W as usize / 2 };
        let ps = self.enc.try_encode(&frame, self.frame as i64 * self.den).map_err(|e| ExportError::Encode(e.to_string()))?;
        let ps = self.packets(ps);
        self.ready.extend(ps);
        self.frame += 1;
        Ok(())
    }
    fn take_packets(&mut self) -> Result<Vec<HardwarePacket>, ExportError> {
        Ok(std::mem::take(&mut self.ready))
    }
    fn finish(&mut self) -> Result<Vec<HardwarePacket>, ExportError> {
        let part = PathBuf::from(format!("{}.part", self.target.display()));
        self.calls.part_only_at_finish.store(part.exists() && !self.target.exists(), Ordering::SeqCst);
        self.calls.finished.fetch_add(1, Ordering::SeqCst);
        let mut ps = std::mem::take(&mut self.ready);
        let tail = self.enc.flush();
        ps.extend(self.packets(tail));
        Ok(ps)
    }
    fn cancel(&mut self) {
        self.calls.cancelled.fetch_add(1, Ordering::SeqCst);
    }
}

fn run(frames: u64, om: &OutputModule, hook: Option<&dyn HardwareVideo>, path: &Path, mut progress: impl FnMut(u64) -> bool) -> Result<Report, ExportError> {
    let (p, cid) = project(frames);
    let settings = RenderSettings { time_span: TimeSpan::LengthOfComp, ..Default::default() };
    let path = path.to_string_lossy().to_string();
    let job = Job {
        project: &p,
        footage: &NoFootage,
        expr: None,
        accel: None,
        comp: cid,
        settings: &settings,
        output: om,
        path: &path,
        sink: None,
        options: JobOptions { hardware: hook, ..Default::default() },
        nested_switches: true,
    };
    export(&job, &mut |pr| progress(pr.done))
}

fn h264() -> OutputModule {
    OutputModule { bitrate_kbps: 4000, ..OutputModule::for_format(OutputFormat::H264) }
}

fn part(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.part", path.display()))
}

/// The frame times the muxer wrote (a B-frame delay without its offsets or edit shows as red
/// arriving early or late).
#[test]
fn hardware_packets_are_muxed_with_the_right_frame_times() {
    for lead in [0u32, 1, 2] {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("hw.mp4");
        let mut stand = Stand::new(&path);
        stand.lead = lead;
        let r = run(12, &h264(), Some(&stand), &path, |_| true).expect("export");

        assert_eq!(r.frames, 12);
        assert!(stand.calls.part_only_at_finish.load(Ordering::SeqCst), "the movie is written beside its target and renamed");
        assert!(path.exists() && !part(&path).exists());
        assert_eq!(stand.calls.cancelled.load(Ordering::SeqCst), 0);
        assert!(r.timing.encoder.contains("Stand-in API (Stand-in GPU), helper 9.9"), "{}", r.timing.encoder);

        let f = effectcraft_media::probe(&path).expect("probe");
        assert_eq!(f.frame_rate.frame_at(f.duration - Tick(1)) + 1, 12, "delay {lead}");
        let pool = MediaPool::new();
        for k in 0..12u64 {
            let img = pool.frame_at(&f, f.frame_rate.tick_of(k as i64)).expect("decode");
            let c = img.data[img.idx(W / 2, H / 2)];
            let want = if k >= RED_FROM { [1.0, 0.0, 0.0] } else { BG };
            let off = (0..3).map(|i| (c[i] - want[i]).abs()).fold(0.0, f32::max);
            assert!(off < 0.08, "delay {lead}: frame {k} centre {c:?}, want {want:?}");
        }
    }
}

#[test]
fn a_hardware_job_that_fails_midway_leaves_no_movie_and_no_part_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("hw.mp4");
    std::fs::write(&path, b"an earlier render").expect("earlier render");
    let mut stand = Stand::new(&path);
    stand.fail = Fail::Submit(60);
    let r = run(80, &h264(), Some(&stand), &path, |_| true);

    match r {
        Err(ExportError::Encode(why)) => assert!(why.contains("the video helper stopped"), "{why}"),
        other => panic!("expected the helper's error, got {other:?}"),
    }
    assert_eq!(std::fs::read(&path).expect("earlier render stays"), b"an earlier render", "the output path is untouched until a movie is whole");
    assert!(!part(&path).exists());
    assert_eq!(stand.calls.cancelled.load(Ordering::SeqCst), 1, "the helper job is closed");
    assert_eq!(stand.calls.finished.load(Ordering::SeqCst), 0);
}

#[test]
fn cancelling_a_hardware_export_closes_the_helper_job_and_removes_the_part_file() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("hw.mp4");
    let stand = Stand::new(&path);
    let r = run(80, &h264(), Some(&stand), &path, |done| done < 40);

    assert!(matches!(r, Err(ExportError::Cancelled)), "{r:?}");
    assert_eq!(stand.calls.cancelled.load(Ordering::SeqCst), 1);
    assert_eq!(stand.calls.finished.load(Ordering::SeqCst), 0);
    assert!(!path.exists() && !part(&path).exists());
}

/// A hook that declines leaves the software encoder's file exactly as without a hook, and the
/// render log's encoder line says why.
#[test]
fn a_declined_hardware_export_is_the_software_export_and_says_why() {
    let dir = tempfile::tempdir().expect("temp dir");
    let (off, auto) = (dir.path().join("off.mp4"), dir.path().join("auto.mp4"));
    let mut stand = Stand::new(&auto);
    stand.declines = Some("No video helper installed".into());

    run(12, &h264(), None, &off, |_| true).expect("export without a hook");
    let r = run(12, &h264(), Some(&stand), &auto, |_| true).expect("export with a declining hook");

    assert_eq!(std::fs::read(&off).expect("off"), std::fs::read(&auto).expect("auto"));
    assert!(!part(&auto).exists());
    assert!(r.timing.encoder.contains("software") && r.timing.encoder.contains("No video helper installed"), "{}", r.timing.encoder);
}

#[test]
fn hardware_is_not_asked_for_formats_other_than_h264() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("hw.mov");
    let stand = Stand::new(&path);
    let om = OutputModule::for_format(OutputFormat::ProRes);
    let r = run(4, &om, Some(&stand), &path, |_| true).expect("export");

    assert_eq!(stand.calls.opened.load(Ordering::SeqCst), 0);
    assert!(r.timing.encoder.contains("H.264 only"), "{}", r.timing.encoder);
}

/// A job whose first packet's offset is not the delay it reported would present its frames at
/// the wrong times without a sign: the export fails and says what it saw.
#[test]
fn a_first_packet_offset_that_differs_from_the_reported_delay_fails_the_export() {
    for (lead, delay_error) in [(1u32, 1u32), (2, 1), (0, 2)] {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("hw.mp4");
        let mut stand = Stand::new(&path);
        stand.lead = lead;
        stand.delay_error = delay_error;
        let r = run(12, &h264(), Some(&stand), &path, |_| true);

        match r {
            Err(ExportError::Encode(why)) => {
                assert!(why.contains(&format!("delay of {} frames", lead + delay_error)) && why.contains("pts - dts must be"), "{why}");
            }
            other => panic!("lead {lead}, delay off by {delay_error}: expected an error, got {other:?}"),
        }
        assert!(!path.exists() && !part(&path).exists());
    }
}
