#include "Simulation/IstanaPolicyInterface.h"

void UIstanaNoOpPolicy::ResetPolicy_Implementation(const FIstanaPolicyContext& Context)
{
    ActiveRunId = Context.RunId;
    LastDecisionStep = INDEX_NONE;
}

void UIstanaNoOpPolicy::ReceiveObservations_Implementation(const FIstanaPolicyObservationBatch& Observations)
{
    if (ActiveRunId.IsValid() && Observations.RunId == ActiveRunId
        && Observations.DecisionStep >= 0 && Observations.DecisionStep > LastDecisionStep)
    {
        LastDecisionStep = Observations.DecisionStep;
    }
}

FIstanaPolicyActionBatch UIstanaNoOpPolicy::ProduceActions_Implementation()
{
    FIstanaPolicyActionBatch Result;
    Result.RunId = ActiveRunId;
    Result.DecisionStep = LastDecisionStep;
    return Result;
}
