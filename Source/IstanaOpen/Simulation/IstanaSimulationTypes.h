#pragma once

#include "CoreMinimal.h"
#include "Engine/World.h"
#include "IstanaSimulationTypes.generated.h"

// Version 1 describes data layout, not the implementation of the simulation.
UENUM(BlueprintType)
enum class EIstanaSensorType : uint8 { AbstractCoverage };

UENUM(BlueprintType)
enum class EIstanaPolicyRole : uint8 { Observer, Blue, Red };

UENUM(BlueprintType)
enum class EIstanaCompletionReason : uint8 { NotCompleted, DurationReached, ScenarioCompleted, Cancelled, Failed };

UENUM(BlueprintType)
enum class EIstanaValidationSeverity : uint8 { Warning, Error };

/** A stable, field-addressable message for both the operator panel and C++ callers. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaValidationIssue
{
    GENERATED_BODY()
    UPROPERTY(BlueprintReadOnly, Category="Validation") FString FieldPath;
    UPROPERTY(BlueprintReadOnly, Category="Validation") EIstanaValidationSeverity Severity = EIstanaValidationSeverity::Error;
    UPROPERTY(BlueprintReadOnly, Category="Validation") FString Message;
};

/** Synthetic conditions. These values do not claim calibration to physical sensors. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaConditions
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Conditions", meta=(ClampMin="0", ClampMax="24")) double TimeOfDayHours = 12.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Conditions", meta=(ClampMin="0", ClampMax="1")) double PrecipitationFraction = 0.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Conditions", meta=(ClampMin="0", ClampMax="1")) double VisibilityFraction = 1.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Conditions") FVector WindVelocityCmPerSecond = FVector::ZeroVector;
};

USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaSwarmConfig
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Swarm") int32 GroupId = INDEX_NONE;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Swarm", meta=(ClampMin="1")) int32 DroneCount = 1;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Swarm") FVector SpawnOriginCm = FVector::ZeroVector;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Swarm", meta=(ClampMin="0")) double SpawnRadiusCm = 0.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Swarm") FName MovementPresetId = TEXT("Stationary");
};

/** One abstract model. Add a new typed model when its implementation exists. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaSensorModelParams
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Model", meta=(ClampMin="0")) double CoverageRadiusCm = 1000.0;
    // Bernoulli probability per eligible object, per sample (not per second).
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Model", meta=(ClampMin="0", ClampMax="1")) double DetectionProbabilityPerSample = 1.0;
    // Probability of one synthetic false observation per sensor sample.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Model", meta=(ClampMin="0", ClampMax="1")) double FalsePositiveProbabilityPerSample = 0.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Model", meta=(ClampMin="0")) double PositionNoiseStdDevCm = 0.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Model", meta=(ClampMin="0")) int32 LatencySteps = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Model") bool bRequiresLineOfSight = false;
};

/** Resolved sensor instance: all model values are copied from a preset. Unit scale only. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaSensorConfig
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Sensor") int32 SensorId = INDEX_NONE;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Sensor") EIstanaSensorType SensorType = EIstanaSensorType::AbstractCoverage;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Sensor") FTransform Transform = FTransform::Identity;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Sensor") FIstanaSensorModelParams Model;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Sensor", meta=(ClampMin="1")) int32 SampleEveryNSteps = 1;
};

/** Value snapshot. The coordinator owns the mutable copy, never the preset asset. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaScenarioConfig
{
    GENERATED_BODY()
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Scenario") int32 SchemaVersion = 1;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Scenario") FName ScenarioId = NAME_None;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Scenario") int32 Seed = 12345;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Scenario", meta=(ClampMin="0.001")) double DurationSeconds = 60.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Scenario", meta=(ClampMin="0.001")) double FixedStepSeconds = 0.05;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Scenario") TSoftObjectPtr<UWorld> Map;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Scenario", meta=(ClampMin="0")) int32 SensorBudget = 1;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Scenario") TArray<FIstanaSwarmConfig> Swarms;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Scenario") FIstanaConditions Conditions;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Scenario") TArray<FIstanaSensorConfig> InitialSensors;
};

/** Evaluator ground truth. Do not put this struct into sensor observation batches. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaDroneState
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="State") int32 DroneId = INDEX_NONE;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="State") int32 GroupId = INDEX_NONE;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="State") FVector PositionCm = FVector::ZeroVector;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="State") FVector VelocityCmPerSecond = FVector::ZeroVector;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="State") bool bActive = true;
};

USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaReportedObservation
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Observation") bool bHasPosition = false;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Observation") FVector PositionCm = FVector::ZeroVector;
};

/** No truth identity or false-positive label: evaluator associations must be stored separately. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaDetectionEvent
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Detection") int32 SensorId = INDEX_NONE;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Detection") int64 SequenceNumber = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Detection") int64 SampleStep = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Detection") int64 DeliveryStep = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Detection") FIstanaReportedObservation Observation;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Detection", meta=(ClampMin="0", ClampMax="1")) double Confidence = 1.0;
};

/** Counts have explicit denominators; missing first detection is not reported as time zero. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaEpisodeMetrics
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Metrics") int64 SensorSampleCount = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Metrics") int64 EligibleObjectSampleCount = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Metrics") int64 TrueDetectionCount = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Metrics") int64 MissedDetectionCount = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Metrics") int64 FalseObservationCount = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Metrics") bool bHasFirstDetection = false;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Metrics") int64 FirstDetectionStep = 0;
};

USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaEpisodeResult
{
    GENERATED_BODY()
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Result") int32 SchemaVersion = 1;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Result") FName ScenarioId = NAME_None;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Result") FGuid RunId;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Result") int32 Seed = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Result") FString SimulationVersion;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Result") FString SensorModelVersion;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Result") int64 ExecutedSteps = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Result") double FixedStepSeconds = 0.05;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Result") FIstanaEpisodeMetrics Metrics;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Result") EIstanaCompletionReason CompletionReason = EIstanaCompletionReason::NotCompleted;
};

USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaPolicyContext
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Policy") FGuid RunId;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Policy") EIstanaPolicyRole Role = EIstanaPolicyRole::Observer;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Policy") int32 PolicySeed = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Policy") double FixedStepSeconds = 0.05;
};

/** The coordinator filters detections before delivery; this is not a world snapshot. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaPolicyObservationBatch
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Policy") FGuid RunId;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Policy") int64 DecisionStep = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Policy") TArray<FIstanaDetectionEvent> Detections;
};

/** V1 supports no-op only. Add typed commands with controller validation in a later schema. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaPolicyActionBatch
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Policy") FGuid RunId;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Policy") int64 DecisionStep = 0;
};
