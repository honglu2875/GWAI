use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GwError {
    UnsupportedInvariant(String),
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
