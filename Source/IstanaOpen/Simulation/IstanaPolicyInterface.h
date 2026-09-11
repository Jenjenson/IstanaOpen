#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "Simulation/IstanaSimulationTypes.h"
#include "IstanaPolicyInterface.generated.h"

UINTERFACE(BlueprintType, Blueprintable)
class ISTANAOPEN_API UIstanaPolicyInterface : public UInterface
{
    GENERATED_BODY()
};

/** Synchronous contract. Invoke Execute_* wrappers to honor Blueprint implementations. */
class ISTANAOPEN_API IIstanaPolicyInterface
{
    GENERATED_BODY()
public:
    UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category="Istana|Policy")
    void ResetPolicy(const FIstanaPolicyContext& Context);

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category="Istana|Policy")
    void ReceiveObservations(const FIstanaPolicyObservationBatch& Observations);

    UFUNCTION(BlueprintCallable, BlueprintNativeEvent, Category="Istana|Policy")
    FIstanaPolicyActionBatch ProduceActions();
};

/** Reference implementation: rejects stale input and produces a correctly stamped no-op. */
UCLASS(BlueprintType, Blueprintable)
class ISTANAOPEN_API UIstanaNoOpPolicy : public UObject, public IIstanaPolicyInterface
{
    GENERATED_BODY()
public:
    virtual void ResetPolicy_Implementation(const FIstanaPolicyContext& Context) override;
    virtual void ReceiveObservations_Implementation(const FIstanaPolicyObservationBatch& Observations) override;
    virtual FIstanaPolicyActionBatch ProduceActions_Implementation() override;

private:
    UPROPERTY(Transient) FGuid ActiveRunId;
    UPROPERTY(Transient) int64 LastDecisionStep = INDEX_NONE;
};
