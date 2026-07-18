use tracegrams::{InitError, Tracegrams};

fn assert_send_sync<T: Send + Sync>(_: &T) {}

#[test]
fn readme_style_initialization_builds_a_shared_handle() -> Result<(), InitError> {
    let mut builder = Tracegrams::builder();
    let parse = builder.stage("parse")?;
    let db = builder.stage("db")?;
    let render = builder.stage("render")?;
    assert_ne!(parse, db);
    assert_ne!(db, render);

    builder.tail_quantile(0.99)?;
    builder.memory_budget_bytes(8 * 1024 * 1024);
    let tracegrams = builder.build()?;
    let cloned = tracegrams.clone();

    assert_send_sync(&tracegrams);
    assert_send_sync(&cloned);
    Ok(())
}

#[test]
fn one_six_and_sixty_four_stage_registries_fit_the_default_budget() {
    for stage_count in [1, 6, 64] {
        let mut builder = Tracegrams::builder();
        for stage in 0..stage_count {
            builder.stage(&format!("stage-{stage}")).unwrap();
        }

        let estimate = builder.estimated_memory().unwrap();
        assert!(estimate.total_bytes() <= 8 * 1024 * 1024);
        builder.build().unwrap();
    }
}

#[test]
fn stage_names_are_non_empty_and_unique() {
    let mut builder = Tracegrams::builder();
    assert_eq!(builder.stage(""), Err(InitError::EmptyStageName));

    builder.stage("db").unwrap();
    assert_eq!(
        builder.stage("db"),
        Err(InitError::DuplicateStageName {
            name: "db".to_owned(),
        })
    );
}

#[test]
fn stage_count_must_be_between_one_and_sixty_four() {
    assert!(matches!(
        Tracegrams::builder().build(),
        Err(InitError::NoStages)
    ));

    let mut builder = Tracegrams::builder();
    for stage in 0..64 {
        builder.stage(&format!("stage-{stage}")).unwrap();
    }
    assert_eq!(
        builder.stage("stage-64"),
        Err(InitError::TooManyStages { maximum: 64 })
    );
}

#[test]
fn tail_quantile_must_be_finite_and_in_range() {
    for quantile in [
        f64::NAN,
        f64::NEG_INFINITY,
        f64::INFINITY,
        -1.0,
        0.0,
        1.000_001,
    ] {
        let mut builder = Tracegrams::builder();
        assert!(matches!(
            builder.tail_quantile(quantile),
            Err(InitError::InvalidTailQuantile { .. })
        ));
    }

    let mut builder = Tracegrams::builder();
    builder.stage("only").unwrap();
    builder.tail_quantile(1.0).unwrap();
    builder.build().unwrap();
}

#[test]
fn memory_estimate_is_itemized_and_pins_the_matrix_formula() {
    let mut builder = Tracegrams::builder();
    for name in ["accept", "parse", "cache", "db", "render", "write"] {
        builder.stage(name).unwrap();
    }

    let estimate = builder.estimated_memory().unwrap();
    assert_eq!(estimate.matrix_bytes(), 366_592);
    assert_eq!(estimate.calibration_bytes(), 6 * 6_000);
    assert!(estimate.online_bytes() > 0);
    assert!(estimate.bounds_bytes() > 0);
    assert!(estimate.stage_metadata_bytes() > 0);
    assert_eq!(estimate.completion_bytes(), 6 * 2 * 8);
    assert!(estimate.diagnostics_bytes() > 0);
    assert!(estimate.recorder_metadata_bytes() > 0);
    assert_eq!(
        estimate.total_bytes(),
        estimate.matrix_bytes()
            + estimate.calibration_bytes()
            + estimate.online_bytes()
            + estimate.bounds_bytes()
            + estimate.stage_metadata_bytes()
            + estimate.completion_bytes()
            + estimate.diagnostics_bytes()
            + estimate.recorder_metadata_bytes()
    );
}

#[test]
fn build_rejects_an_estimate_above_the_budget() {
    let mut builder = Tracegrams::builder();
    builder.stage("only").unwrap();
    let required = builder.estimated_memory().unwrap().total_bytes();
    builder.memory_budget_bytes(required - 1);

    assert!(matches!(
        builder.build(),
        Err(InitError::MemoryBudgetExceeded {
            required: actual_required,
            budget,
        }) if actual_required == required && budget == required - 1
    ));
}
