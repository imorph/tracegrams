//! Collect-freeze-diagnose flow with every checkpoint written at its call site.

use std::error::Error;
use std::time::Duration;

use tracegrams::{Classification, DiagnoseConfig, FreezeCriteria, Outcome, Tracegrams};

const CALIBRATION_REQUESTS: u64 = 100;

fn main() -> Result<(), Box<dyn Error>> {
    let mut builder = Tracegrams::builder();
    let parse = builder.stage("parse")?;
    let database = builder.stage("database")?;
    let response = builder.stage("response")?;
    builder.tail_quantile(0.99)?;
    let tracegrams = builder.build()?;

    for _ in 0..CALIBRATION_REQUESTS {
        let mut context = tracegrams.start_manual();
        tracegrams.record_elapsed(&mut context, parse, Duration::from_millis(1));
        tracegrams.record_elapsed(&mut context, database, Duration::from_millis(3));
        tracegrams.finish_manual(
            context,
            response,
            Duration::from_millis(1),
            Outcome::Success,
        );
    }
    tracegrams.try_freeze_calibration(FreezeCriteria::all_stages(CALIBRATION_REQUESTS))?;

    let before = tracegrams.snapshot_relaxed();

    let mut context = tracegrams.start_manual();
    tracegrams.record_elapsed(&mut context, parse, Duration::from_micros(100));
    tracegrams.record_elapsed(&mut context, database, Duration::from_millis(30));
    tracegrams.finish_manual(
        context,
        response,
        Duration::from_micros(100),
        Outcome::Success,
    );

    let incident = tracegrams.snapshot_relaxed().delta(&before)?;
    let report = incident.diagnose(database, &DiagnoseConfig::experimental_defaults())?;
    assert_eq!(report.classification(), Classification::Onset);

    Ok(())
}
