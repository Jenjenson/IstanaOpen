#pragma once

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "Simulation/IstanaSimulationTypes.h"
#include "IstanaSimulationPresets.generated.h"

/** Content Browser: Miscellaneous > Data Asset > IstanaScenarioDataAsset. */
UCLASS(BlueprintType)
class ISTANAOPEN_API UIstanaScenarioDataAsset : public UDataAsset
{
    GENERATED_BODY()
public:
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Scenario") FText Description;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Scenario") FIstanaScenarioConfig Config;

    UFUNCTION(BlueprintPure, Category="Istana|Simulation")
    FIstanaScenarioConfig CreateRuntimeConfig() const { return Config; }
};

/** A reusable model template, never a live sensor or its placement. */
UCLASS(BlueprintType)
class ISTANAOPEN_API UIstanaSensorPresetDataAsset : public UDataAsset
{
    GENERATED_BODY()
public:
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Sensor") FText Description;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Sensor") EIstanaSensorType SensorType = EIstanaSensorType::AbstractCoverage;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Sensor") FIstanaSensorModelParams Model;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Sensor", meta=(ClampMin="1")) int32 SampleEveryNSteps = 1;

    UFUNCTION(BlueprintPure, Category="Istana|Simulation")
    FIstanaSensorConfig CreateSensorConfig(int32 SensorId, const FTransform& Transform) const
    {
        FIstanaSensorConfig Result;
        Result.SensorId = SensorId;
        Result.Transform = Transform;
        Result.SensorType = SensorType;
        Result.Model = Model;
        Result.SampleEveryNSteps = SampleEveryNSteps;
        return Result;
    }
};
