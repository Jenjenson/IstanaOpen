#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "Simulation/IstanaSimulationTypes.h"
#include "IstanaSimulationValidation.generated.h"

/** Pure structural validation: no world loading, mutation, or simulation side effects. */
UCLASS()
class ISTANAOPEN_API UIstanaSimulationValidation : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:
    // All validators replace Issues. True means no errors; warnings may remain.
    UFUNCTION(BlueprintCallable, Category="Istana|Validation")
    static bool ValidateScenario(const FIstanaScenarioConfig& Config, TArray<FIstanaValidationIssue>& Issues);

    UFUNCTION(BlueprintCallable, Category="Istana|Validation")
    static bool ValidateDroneStates(const FIstanaScenarioConfig& Config, const TArray<FIstanaDroneState>& States, TArray<FIstanaValidationIssue>& Issues);

    UFUNCTION(BlueprintCallable, Category="Istana|Validation")
    static bool ValidateSensor(const FIstanaSensorConfig& Sensor, TArray<FIstanaValidationIssue>& Issues);

    UFUNCTION(BlueprintCallable, Category="Istana|Validation")
    static bool ValidateDetection(const FIstanaDetectionEvent& Event, const TArray<int32>& KnownSensorIds, TArray<FIstanaValidationIssue>& Issues);

    UFUNCTION(BlueprintCallable, Category="Istana|Validation")
    static bool ValidateEpisodeResult(const FIstanaEpisodeResult& Result, TArray<FIstanaValidationIssue>& Issues);

    UFUNCTION(BlueprintCallable, Category="Istana|Validation")
    static bool ValidateActionBatch(const FIstanaPolicyActionBatch& Actions, const FIstanaPolicyContext& Context, int64 ExpectedDecisionStep, TArray<FIstanaValidationIssue>& Issues);

    // In-memory example only. Assign a fictional map before launch validation succeeds.
    UFUNCTION(BlueprintPure, Category="Istana|Simulation")
    static FIstanaScenarioConfig MakeExampleScenario();
};
