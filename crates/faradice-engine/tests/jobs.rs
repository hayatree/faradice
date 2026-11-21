//! Jobs: event order, cancellation, panics, partial results, and agreement
//! with blocking runs.

#![allow(missing_docs)]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use faradice_engine::{EngineError, Job, JobEvent, JobState, Project, RunRequest};
use faradice_io::config::{Config, Preset};
use faradice_io::result::{Completeness, NetRecord, ResultFile};

fn project() -> Arc<Project> {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/three_wires.faradice.toml");
    Arc::new(Project::load(&p, None, None).unwrap())
}

fn request(preset: Preset) -> RunRequest {
    let mut config = Config::default();
    config.accuracy.preset = preset;
    config.run.seed = 3;
    RunRequest {
        config,
        ..RunRequest::default()
    }
}

/// Runs a job and collects *every* event until the sink is dropped, so that
/// duplicate terminals or late progress reports would be seen.
fn collect(
    req: RunRequest,
    on_event: impl Fn(&Job, &JobEvent),
) -> (Vec<JobEvent>, Result<(), String>) {
    let (tx, rx) = mpsc::channel();
    let job = Job::start(project(), req, move |e| {
        let _ = tx.send(e);
    });
    let mut events = Vec::new();
    loop {
        match rx.recv_timeout(Duration::from_secs(120)) {
            Ok(e) => {
                on_event(&job, &e);
                events.push(e);
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => panic!("job did not finish"),
        }
    }
    (events, job.join())
}

fn assert_one_terminal_last(events: &[JobEvent]) {
    let terminals = events.iter().filter(|e| e.state().is_terminal()).count();
    assert_eq!(terminals, 1, "exactly one terminal event");
    assert!(
        events.last().unwrap().state().is_terminal(),
        "terminal event is last"
    );
}

fn bits(r: &NetRecord) -> Vec<(String, u64, u64)> {
    r.sinks
        .iter()
        .map(|s| (s.sink.clone(), s.sum.to_bits(), s.sum_sq.to_bits()))
        .collect()
}

#[test]
fn events_are_ordered_and_match_a_blocking_run() {
    let (events, joined) = collect(request(Preset::Fast), |_, _| {});
    joined.unwrap();
    assert_one_terminal_last(&events);
    let JobEvent::Started { targets, .. } = &events[0] else {
        panic!("first event must be Started, got {:?}", events[0]);
    };
    assert_eq!(targets.len(), 4);
    let last = events.last().unwrap();
    let JobEvent::Finished(out) = last else {
        panic!("last event must be Finished, got {last:?}");
    };
    assert!(!out.cancelled);
    assert_eq!(out.result.completeness, Completeness::Complete);
    assert!(matches!(
        last.state(),
        JobState::Succeeded | JobState::NotConverged
    ));
    for i in 0..targets.len() {
        let mine: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                JobEvent::Progress(p) if p.target == i => Some(p),
                _ => None,
            })
            .collect();
        assert_eq!(mine.iter().filter(|p| p.finished).count(), 1);
        assert!(
            mine.last().unwrap().finished,
            "final report is the last of its net"
        );
    }
    let blocking = faradice_engine::run(&project(), &request(Preset::Fast), &()).unwrap();
    assert_eq!(blocking.result.targets.len(), out.result.targets.len());
    for (a, b) in blocking.result.targets.iter().zip(&out.result.targets) {
        assert_eq!(
            (&a.net, a.walks, a.converged),
            (&b.net, b.walks, b.converged)
        );
        assert_eq!(bits(a), bits(b));
        assert_eq!(a.ground.sum.to_bits(), b.ground.sum.to_bits());
    }
}

#[test]
fn cancel_yields_partial_results_that_round_trip() {
    let (events, joined) = collect(request(Preset::Reference), |job, e| {
        if matches!(e, JobEvent::Progress(_)) {
            job.cancel();
        }
    });
    joined.unwrap();
    assert_one_terminal_last(&events);
    let last = events.last().unwrap();
    assert_eq!(last.state(), JobState::Cancelled);
    let JobEvent::Finished(out) = last else {
        unreachable!()
    };
    assert!(out.cancelled);
    assert_eq!(out.result.completeness, Completeness::Cancelled);
    let dir = std::env::temp_dir().join(format!("faradice-engine-test-{}", std::process::id()));
    faradice_engine::write_outputs(out, &dir, "t").unwrap();
    let back = ResultFile::read(&dir.join("result.json")).unwrap();
    assert_eq!(back.completeness, Completeness::Cancelled);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn complete_results_omit_completeness() {
    let out = faradice_engine::run(&project(), &request(Preset::Fast), &()).unwrap();
    let json = serde_json::to_string(&out.result).unwrap();
    assert!(!json.contains("completeness"));
}

#[test]
fn partial_snapshots_are_available_while_running() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let s = Arc::clone(&seen);
    let (events, joined) = collect(request(Preset::Default), move |job, e| {
        if matches!(e, JobEvent::Progress(_)) {
            if let Some(p) = job.partial() {
                s.lock()
                    .unwrap()
                    .push((job.version(), p.result.completeness, p.report.nets.len()));
            }
        }
    });
    joined.unwrap();
    assert_one_terminal_last(&events);
    let seen = seen.lock().unwrap();
    assert!(!seen.is_empty());
    assert!(
        seen.iter()
            .all(|(v, c, n)| *v > 0 && *c == Completeness::Partial && *n > 0)
    );
    assert!(seen.windows(2).all(|w| w[0].0 <= w[1].0));
}

#[test]
fn invalid_request_fails_without_started() {
    let mut req = request(Preset::Fast);
    req.config.extract.nets = vec!["no_such_net".into()];
    let (events, joined) = collect(req, |_, _| {});
    joined.unwrap();
    assert_eq!(events.len(), 1);
    let JobEvent::Failed(e) = &events[0] else {
        panic!("expected Failed")
    };
    assert!(matches!(**e, EngineError::Io(_)), "typed error: {e:?}");
}

#[test]
fn panicking_sink_becomes_failed() {
    let (tx, rx) = mpsc::channel();
    let job = Job::start(project(), request(Preset::Fast), move |e| {
        let progress = matches!(e, JobEvent::Progress(_));
        let _ = tx.send(e);
        assert!(!progress, "sink panics on progress");
    });
    let events: Vec<JobEvent> = rx.iter().collect();
    job.join().unwrap();
    assert_one_terminal_last(&events);
    let JobEvent::Failed(e) = events.last().unwrap() else {
        panic!("expected Failed")
    };
    assert!(matches!(**e, EngineError::Panic(_)), "{e:?}");
}

/// Cancelled before the solver starts: no stopping distance yet, still a
/// valid result file.
#[test]
fn cancelled_before_start_round_trips() {
    struct Stop;
    impl faradice_frw::Observer for Stop {
        fn cancelled(&self) -> bool {
            true
        }
    }
    let out = faradice_engine::run(&project(), &request(Preset::Fast), &Stop).unwrap();
    assert!(out.cancelled);
    assert!(out.result.targets.is_empty());
    assert!(out.result.eps_stop.is_nan());
    let dir = std::env::temp_dir().join(format!("faradice-engine-pre-{}", std::process::id()));
    faradice_engine::write_outputs(&out, &dir, "t").unwrap();
    let back = ResultFile::read(&dir.join("result.json")).unwrap();
    assert_eq!(back.completeness, Completeness::Cancelled);
    assert!(back.eps_stop.is_nan());
    std::fs::remove_dir_all(&dir).unwrap();
}
