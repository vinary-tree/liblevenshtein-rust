//! Ordered cost algebras used by bounded dynamic programs.
//!
//! This module deliberately exposes a monoid, not a semiring. Dynamic-programming
//! paths combine successive step costs with [`crate::cost::CostMonoid::combine`], compare totals
//! with [`crate::cost::CostMonoid::compare`], and always choose the smaller total with
//! [`crate::cost::CostMonoid::select`]. There is no caller-configurable additive choice, Kleene
//! star, or division operation.
//!
//! The lawful carrier domain is documented by [`crate::cost::CostMonoid`]. In particular, the
//! floating-point implementations are intended for finite, non-negative costs plus
//! the distinguished positive-infinity value [`crate::cost::CostMonoid::TOP`].

mod bottleneck;
mod monoid;
mod scale;
mod subsumption;
mod unit;
mod weighted;

pub use bottleneck::BottleneckCost;
pub use monoid::CostMonoid;
pub use scale::{CostScale, ScaleError};
pub use unit::UnitCost;
pub use weighted::WeightedCost;

pub(crate) use subsumption::{subsumes_with, SubsumptionMode};
