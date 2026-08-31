# Tracegrams

Tracking Tail Latency Propagation Without Storing Traces

![Tracegrams latency propagation demo](https://raw.githubusercontent.com/imorph/tracegrams/main/assets/tracegrams-demo.gif)

Histograms are cheap to maintain and great at showing aggregate latency, but they do not explain why requests got slow. Traces preserve the full causal story of a single request, but at scale they are expensive, can hurt the hot path and are hard to aggregate, sample, and retain.

Can we get the best of both worlds without breaking the bank? Maybe.

Tracegrams are a lightweight measurement primitive: each request carries a tiny piece of latency state, and checkpoints along the request path update transition histograms. The aim is to see where p99 latency is born, where it gets amplified, and where it is carried downstream, all online, with minimal overhead for the application itself.

I'll show a Rust prototype, synthetic stress tests with known ground truth, a negative-control scenario, nanoseconds-per-checkpoint overhead numbers, and a path from prototype to a crate that can be used in async Rust services and evaluated on real OSS benchmarks.

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  https://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or
  https://opensource.org/licenses/MIT)

at your option.

#### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
