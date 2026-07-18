//! Compile-time checks for the compatibility facades retained by the source
//! tree refactor.  This is an integration test deliberately using the crate as
//! an external caller would.

#![allow(deprecated)]

use gw_pn::algebra::{Coeff, RatFun};
use gw_pn::error::GwError;
use gw_pn::givental::{
    CoefficientSemisimpleCohftProvider, FactoredProjectiveSpaceProvider, GwTarget,
    ProjectiveSpaceProvider, SemisimpleCohftProvider,
};
use gw_pn::{Insertion, InvariantRequest, InvariantResult, Truncation};

fn root_request_is_space_request(
    request: InvariantRequest,
) -> gw_pn::spaces::projective_space::InvariantRequest {
    request
}

fn historical_theory_is_space_theory(
    theory: gw_pn::theory::ProjectiveSpaceTheory,
) -> gw_pn::spaces::projective_space::ProjectiveSpaceTheory {
    theory
}

fn historical_resolvent_is_space_resolvent(
    request: gw_pn::resolvent::ResolventRequest,
) -> gw_pn::spaces::projective_space::ResolventRequest {
    request
}

fn historical_projective_evaluator_is_space_evaluator(
    evaluator: gw_pn::constraints::virasoro::ProjectiveSpaceEvaluator,
) -> gw_pn::spaces::projective_space::ProjectiveSpaceEvaluator {
    evaluator
}

fn historical_product_evaluator_is_space_evaluator(
    evaluator: gw_pn::constraints::virasoro::ProductProjectiveEvaluator,
) -> gw_pn::spaces::product_projective::ProductProjectiveEvaluator {
    evaluator
}

fn historical_bundle_evaluator_is_space_evaluator(
    evaluator: gw_pn::constraints::virasoro::ProjectiveBundleEvaluator,
) -> gw_pn::spaces::projective_bundle::ProjectiveBundleEvaluator {
    evaluator
}

fn historical_local_evaluator_is_space_evaluator(
    evaluator: gw_pn::constraints::virasoro::NegativeSplitCompletionEvaluator,
) -> gw_pn::spaces::negative_split_projective::NegativeSplitCompletionEvaluator {
    evaluator
}

fn historical_completion_is_space_completion(
    completion: gw_pn::theory::NegativeSplitProjectiveCompletion,
) -> gw_pn::spaces::negative_split_projective::NegativeSplitProjectiveCompletion {
    completion
}

fn negative_split_provider_module_is_flattened_provider(
    provider: gw_pn::spaces::negative_split_projective::provider::TwistedProjectiveSpaceProvider,
) -> gw_pn::spaces::negative_split_projective::TwistedProjectiveSpaceProvider {
    provider
}

fn negative_split_twist_module_is_flattened_twist(
    twist: gw_pn::spaces::negative_split_projective::twist::NegativeSplitBundleTwist,
) -> gw_pn::spaces::negative_split_projective::NegativeSplitBundleTwist {
    twist
}

fn historical_series_master(
    request: &gw_pn::spaces::projective_space::SeriesRequest,
    provider: ProjectiveSpaceProvider,
) -> Result<Option<gw_pn::spaces::projective_space::SeriesResult>, GwError> {
    gw_pn::givental::compute_series_master_with_provider(request, provider)
}

fn historical_packed_resolvent(
    request: &gw_pn::resolvent::ResolventRequest,
    provider: ProjectiveSpaceProvider,
) -> Result<gw_pn::resolvent::ResolventResult, GwError> {
    gw_pn::givental::compute_packed_resolvent_with_provider(
        request,
        provider,
        "compatibility-test",
        "compatibility-test",
        Ok::<RatFun, GwError>,
    )
}

fn historical_factored_packed_resolvent(
    request: &gw_pn::resolvent::ResolventRequest,
    provider: FactoredProjectiveSpaceProvider,
) -> Result<gw_pn::resolvent::ResolventResult<gw_pn::factored::FactoredRatFun>, GwError> {
    gw_pn::givental::compute_packed_resolvent_with_coeff_provider(
        request,
        provider,
        "compatibility-test",
        "compatibility-test",
        Ok::<gw_pn::factored::FactoredRatFun, GwError>,
    )
}

fn root_truncation_is_givental(truncation: Truncation) -> gw_pn::givental::Truncation {
    truncation
}

fn givental_truncation_is_projective_api(
    truncation: gw_pn::givental::Truncation,
) -> gw_pn::spaces::projective_space::api::Truncation {
    truncation
}

fn projective_api_truncation_is_space(
    truncation: gw_pn::spaces::projective_space::api::Truncation,
) -> gw_pn::spaces::projective_space::Truncation {
    truncation
}

fn historical_target_method<T: GwTarget>(
    target: &T,
    class: &gw_pn::geometry::CohomologyClass,
) -> Result<Vec<gw_pn::series::QSeries>, GwError> {
    target.insertion_vector(class, 0)
}

fn assert_legacy_coefficient_view<C, P>()
where
    C: Coeff,
    P: SemisimpleCohftProvider<C, Insertion = Insertion>
        + CoefficientSemisimpleCohftProvider<C, Insertion = Insertion>,
{
}

#[test]
#[allow(deprecated)]
fn historical_paths_remain_source_compatible() {
    let _ = root_request_is_space_request;
    let _ = historical_theory_is_space_theory;
    let _ = historical_resolvent_is_space_resolvent;
    let _ = historical_projective_evaluator_is_space_evaluator;
    let _ = historical_product_evaluator_is_space_evaluator;
    let _ = historical_bundle_evaluator_is_space_evaluator;
    let _ = historical_local_evaluator_is_space_evaluator;
    let _ = historical_completion_is_space_completion;
    let _ = negative_split_provider_module_is_flattened_provider;
    let _ = negative_split_twist_module_is_flattened_twist;
    let _ = historical_series_master;
    let _ = historical_packed_resolvent;
    let _ = historical_factored_packed_resolvent;
    let _ = historical_target_method::<gw_pn::givental::ProjectiveTarget>;

    let truncation = root_truncation_is_givental(Truncation { z_order: 3 });
    let truncation = givental_truncation_is_projective_api(truncation);
    let truncation = projective_api_truncation_is_space(truncation);
    assert_eq!(truncation.z_order, 3);
    assert_eq!(
        gw_pn::MAX_UNSTABLE_DIVISOR_RECURSION_TOTAL_PSI,
        gw_pn::core::moduli::MAX_UNSTABLE_DIVISOR_RECURSION_TOTAL_PSI
    );

    let _: fn(&InvariantRequest) -> Result<InvariantResult, GwError> = gw_pn::givental::compute;
    let _: fn(&InvariantRequest) -> Result<InvariantResult, GwError> =
        gw_pn::spaces::projective_space::compute_givental;

    assert_legacy_coefficient_view::<RatFun, ProjectiveSpaceProvider>();
    assert_legacy_coefficient_view::<
        gw_pn::factored::FactoredRatFun,
        FactoredProjectiveSpaceProvider,
    >();
    let _ = gw_pn::givental::projective_space_j_calibration;
    let _ = gw_pn::givental::projective_space_descendant_s_matrix;
    let _ = std::mem::size_of::<gw_pn::givental::ProjectiveSpaceJCalibration>();

    fn product_alias(
        insertion: gw_pn::givental::product::ProductInsertion,
    ) -> gw_pn::spaces::product_projective::ProductInsertion {
        insertion
    }
    fn bundle_alias(
        insertion: gw_pn::givental::bundle::BundleInsertion,
    ) -> gw_pn::spaces::projective_bundle::BundleInsertion {
        insertion
    }
    let _ = product_alias;
    let _ = bundle_alias;
}
