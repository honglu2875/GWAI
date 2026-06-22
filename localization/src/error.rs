use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GwError {
    DimensionMismatch { expected: isize, actual: usize },
    UnsupportedInvariant(String),
    MissingHodgeIntegralBackend,
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
            GwError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "dimension mismatch: virtual dimension is {expected}, insertion degree is {actual}"
                )
            }
            GwError::UnsupportedInvariant(msg) => write!(f, "unsupported invariant: {msg}"),
            GwError::MissingHodgeIntegralBackend => write!(f, "missing Hodge-integral backend"),
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
