pub mod aggregation;
pub mod data_pipeline;
pub mod tee_interface;

pub use aggregation::{federated_averaging, multi_krum};
pub use data_pipeline::{PacketView, XdpIngress};
pub use tee_interface::{AggregationAlgorithm, ComputationParams, InMemoryTee, TeeError, TeeGuard};
