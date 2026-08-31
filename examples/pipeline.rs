//! Collect-freeze-diagnose flow with request recording factored into a helper.

use std::error::Error;
use std::time::Duration;

use tracegrams::{Classification, DiagnoseConfig, FreezeCriteria, Outcome, StageId, Tracegrams};

const CALIBRATION_REQUESTS: u64 = 100;

fn record_manual_request(tracegrams: &Tracegrams, stages: [StageId; 3], elapsed: [Duration; 3]) {
    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, stages[0], elapsed[0]);
    tracegrams.record_elapsed(&mut context, stages[1], elapsed[1]);
    tracegrams.finish_manual(context, stages[2], elapsed[2], Outcome::Success);
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut builder = Tracegrams::builder();
    let parse = builder.stage("parse")?;
    let database = builder.stage("database")?;
    let response = builder.stage("response")?;
    builder.tail_quantile(0.99)?;
    let tracegrams = builder.build()?;
    let stages = [parse, database, response];

    // Manual timing makes the example deterministic. A production handler uses
    // start/mark/finish around the same stage boundaries.
    for _ in 0..CALIBRATION_REQUESTS {
        record_manual_request(
            &tracegrams,
            stages,
            [
                Duration::from_millis(1),
                Duration::from_millis(3),
                Duration::from_millis(1),
            ],
        );
    }
    assert!(
        tracegrams
            .calibration_readiness(CALIBRATION_REQUESTS)
            .is_ready()
    );
    tracegrams.try_freeze_calibration(FreezeCriteria::all_stages(CALIBRATION_REQUESTS))?;

    let before = tracegrams.snapshot_relaxed();
    record_manual_request(
        &tracegrams,
        stages,
        [
            Duration::from_micros(100),
            Duration::from_millis(30),
            Duration::from_micros(100),
        ],
    );
    let after = tracegrams.snapshot_relaxed();

    let incident = after.delta(&before)?;
    let policy = DiagnoseConfig::experimental_defaults();
    let report = incident.diagnose(database, &policy)?;
    assert_eq!(report.classification(), Classification::Onset);
    let _display_report = report.to_string();

    Ok(())
}
