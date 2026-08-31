# Development tasks used locally and in CI.
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Keep in sync with rust-version and the CI MSRV job.
msrv := "1.85.0"

# List recipes.
default:
    @just --list

# Install additional toolchains and Cargo tools.
setup:
    rustup toolchain install nightly
    rustup toolchain install {{ msrv }}
    cargo install --locked cargo-docs-rs cargo-deny zizmor

# Apply formatting.
fmt:
    cargo fmt

# Check formatting.
fmt-check:
    cargo fmt --check

# Run Clippy with warnings denied.
clippy:
    cargo clippy --locked --all-targets --all-features -- -D warnings

# Run tests, benchmarks in test mode, and doctests.
test:
    cargo test --locked --all-features --all-targets
    cargo test --locked --all-features --doc

# Build and verify the packaged crate.
package-check:
    cargo package --allow-dirty --locked

# Build documentation using the docs.rs configuration.
[unix]
doc:
    RUSTDOCFLAGS="-D warnings" cargo +nightly docs-rs

[windows]
doc:
    $env:RUSTDOCFLAGS="-D warnings"; cargo +nightly docs-rs

# Check the exact minimum supported Rust version.
msrv-check:
    cargo +{{ msrv }} check --locked --all-features --all-targets

# Check minimum dependency versions.
[unix]
minimal-versions:
    #!/usr/bin/env bash
    set -euo pipefail
    lockfile_backup="$(mktemp)"
    cp Cargo.lock "$lockfile_backup"
    trap 'cp "$lockfile_backup" Cargo.lock; rm -f "$lockfile_backup"' EXIT
    cargo +nightly generate-lockfile -Zminimal-versions
    cargo +nightly check --locked --all-features --all-targets

[windows]
minimal-versions:
    $ErrorActionPreference = 'Stop'; $lockfileBackup = [System.IO.Path]::GetTempFileName(); Copy-Item Cargo.lock $lockfileBackup; try { cargo +nightly generate-lockfile -Zminimal-versions; if ($LASTEXITCODE -ne 0) { throw "cargo generate-lockfile failed with exit code $LASTEXITCODE" }; cargo +nightly check --locked --all-features --all-targets; if ($LASTEXITCODE -ne 0) { throw "cargo check failed with exit code $LASTEXITCODE" } } finally { Copy-Item $lockfileBackup Cargo.lock -Force; Remove-Item $lockfileBackup -Force }

# Check dependency licenses, bans, and sources.
deny:
    cargo deny check bans licenses sources

# Audit GitHub Actions workflows.
actions-check:
    zizmor .github/workflows

# Check RustSec advisories.
deny-advisories:
    cargo deny check advisories

# Test the latest dependency versions.
[unix]
test-latest-deps:
    #!/usr/bin/env bash
    set -euo pipefail
    lockfile_backup="$(mktemp)"
    cp Cargo.lock "$lockfile_backup"
    trap 'cp "$lockfile_backup" Cargo.lock; rm -f "$lockfile_backup"' EXIT
    cargo update
    just test

[windows]
test-latest-deps:
    $ErrorActionPreference = 'Stop'; $lockfileBackup = [System.IO.Path]::GetTempFileName(); Copy-Item Cargo.lock $lockfileBackup; try { cargo update; if ($LASTEXITCODE -ne 0) { throw "cargo update failed with exit code $LASTEXITCODE" }; just test; if ($LASTEXITCODE -ne 0) { throw "just test failed with exit code $LASTEXITCODE" } } finally { Copy-Item $lockfileBackup Cargo.lock -Force; Remove-Item $lockfileBackup -Force }

# Run benchmarks.
bench:
    cargo bench

# Run all blocking checks.
all: fmt-check clippy test package-check doc msrv-check minimal-versions deny actions-check
