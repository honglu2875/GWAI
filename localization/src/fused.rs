//! Small fused arithmetic kernels used by series and graph contractions.
//!
//! The public coefficient types still own their algebra, but hot containers
//! should call these helpers instead of spelling `acc = acc + a * b` inline.
//! That keeps future coefficient-specific in-place kernels localized.

use crate::algebra::Coeff;

pub(crate) fn add_assign<C: Coeff>(target: &mut C, rhs: &C) {
    target.add_assign(rhs);
}

pub(crate) fn add_product_assign<C: Coeff>(target: &mut C, left: &C, right: &C) {
    target.add_product_assign(left, right);
}
