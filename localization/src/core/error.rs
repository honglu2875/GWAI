//! Public error boundary for mathematical support and finite-work failures.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GwError {
    UnsupportedInvariant(String),
    /// A target/backend combination is understood, but a named mathematical
    /// feature is not implemented for the witnessed case.
    UnsupportedFeature {
        target: String,
        feature: String,
        witness: String,
    },
    /// A checked finite-work or retained-state boundary was exceeded. Each
    /// call site stops before the guarded operation or additional retention.
    ResourceLimit {
        operation: String,
        requested: usize,
        limit: usize,
    },
    NonSemisimplePoint,
    TruncationTooLow,
    NonFiniteLimit(String),
    ConventionMismatch(String),
    AlgebraFailure(String),
    ValidationFailure(String),
    ParseError(String),
}

impl fmt::Display for GwError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GwError::UnsupportedInvariant(msg) => write!(f, "unsupported invariant: {msg}"),
            GwError::UnsupportedFeature {
                target,
                feature,
                witness,
            } => write!(f, "unsupported {feature} for {target}: {witness}"),
            GwError::ResourceLimit {
                operation,
                requested,
                limit,
            } => write!(
                f,
                "resource limit: {operation} requested {requested}, limit {limit}"
            ),
            GwError::NonSemisimplePoint => write!(f, "non-semisimple Frobenius point"),
            GwError::TruncationTooLow => write!(f, "requested truncation is too low"),
            GwError::NonFiniteLimit(msg) => write!(f, "non-finite limit: {msg}"),
            GwError::ConventionMismatch(msg) => write!(f, "convention mismatch: {msg}"),
            GwError::AlgebraFailure(msg) => write!(f, "algebra failure: {msg}"),
            GwError::ValidationFailure(msg) => write!(f, "validation failure: {msg}"),
            GwError::ParseError(msg) => write!(f, "parse error: {msg}"),
        }
    }
}

impl std::error::Error for GwError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_errors_preserve_machine_readable_fields() {
        let resource = GwError::ResourceLimit {
            operation: "exact Novikov rays".to_string(),
            requested: 65,
            limit: 64,
        };
        assert!(matches!(
            resource,
            GwError::ResourceLimit {
                requested: 65,
                limit: 64,
                ..
            }
        ));

        let feature = GwError::UnsupportedFeature {
            target: "bundle".to_string(),
            feature: "generalized mirror normalization".to_string(),
            witness: "higher-primary z^-1 coordinate".to_string(),
        };
        assert!(feature.to_string().contains("higher-primary z^-1"));
    }
}
