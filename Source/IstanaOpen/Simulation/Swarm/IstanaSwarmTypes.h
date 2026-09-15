#pragma once

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "IstanaSwarmTypes.generated.h"

UENUM(BlueprintType)
enum class EIstanaSwarmCommandType : uint8
{
    Hold,
    FollowWaypoints,
    SetCruiseSpeed,
    SetSpacing,
    Stop
};

/** Kinematic tuning in cm and seconds. Defaults are synthetic, not flight calibration. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaSwarmSettings
{
    GENERATED_BODY()
    // Bounded 3D grid search; smaller cells resolve narrower passages at greater cost.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Navigation", meta=(ClampMin="25"))
    double NavigationCellSizeCm = 200.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Navigation", meta=(ClampMin="0"))
    double NavigationClearanceCm = 80.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Navigation", meta=(ClampMin="100", ClampMax="100000"))
    int32 MaxNavigationNodes = 12000;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Motion")
    double DroneRadiusCm = 20.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Motion")
    double MaxSpeedCmPerSecond = 600.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Motion")
    double CruiseSpeedCmPerSecond = 300.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Motion")
    double MaxAccelerationCmPerSecondSquared = 200.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Motion")
    double MaxTurnDegreesPerSecond = 90.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Motion")
    double ResponseSeconds = 0.5;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Motion")
    double ArrivalRadiusCm = 120.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Boids")
    double NeighborRadiusCm = 600.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Boids")
    double SpacingCm = 140.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Boids")
    double SeparationWeight = 1.5;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Boids")
    double AlignmentWeight = 0.25;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Boids")
    double CohesionWeight = 0.15;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Limits", meta=(ClampMin="1", ClampMax="256"))
    int32 MaxDrones = 128;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Limits", meta=(ClampMin="1", ClampMax="10000"))
    int32 SpawnAttemptsPerDrone = 1000;
};

UCLASS(BlueprintType)
class ISTANAOPEN_API UIstanaSwarmMovementPreset : public UDataAsset
{
    GENERATED_BODY()
public:
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Swarm")
    FIstanaSwarmSettings Settings;
};

/** Local controller command, deliberately separate from the schema-1 policy no-op. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaSwarmCommand
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Command") FGuid RunId;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Command") int64 DecisionStep = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Command") int64 SequenceNumber = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Command") int32 GroupId = INDEX_NONE;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Command") EIstanaSwarmCommandType Type = EIstanaSwarmCommandType::Hold;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Command") TArray<FVector> WaypointsCm;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Command") bool bLoop = false;
    // Objective following accepts obstructed destinations and approaches reachable space.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Command") bool bAllowPartialPath = false;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Command") double CruiseSpeedCmPerSecond = 300.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Command") double SpacingCm = 140.0;
};

USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaSwarmGroupStatus
{
    GENERATED_BODY()
    UPROPERTY(BlueprintReadOnly, Category="Swarm") int32 GroupId = INDEX_NONE;
    UPROPERTY(BlueprintReadOnly, Category="Swarm") FVector CentroidCm = FVector::ZeroVector;
    UPROPERTY(BlueprintReadOnly, Category="Swarm") FVector TargetCm = FVector::ZeroVector;
    UPROPERTY(BlueprintReadOnly, Category="Swarm") int32 WaypointIndex = INDEX_NONE;
    UPROPERTY(BlueprintReadOnly, Category="Swarm") bool bRouteCompleted = false;
    UPROPERTY(BlueprintReadOnly, Category="Swarm") bool bNavigationBlocked = false;
    UPROPERTY(BlueprintReadOnly, Category="Swarm") bool bHasPartialPath = false;
    UPROPERTY(BlueprintReadOnly, Category="Swarm") EIstanaSwarmCommandType Mode = EIstanaSwarmCommandType::Hold;
};

USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaSwarmDiagnostics
{
    GENERATED_BODY()
    UPROPERTY(BlueprintReadOnly, Category="Diagnostics") int64 ExecutedSteps = 0;
    UPROPERTY(BlueprintReadOnly, Category="Diagnostics") double SimulatedSeconds = 0.0;
    UPROPERTY(BlueprintReadOnly, Category="Diagnostics") double TotalPathLengthCm = 0.0;
    // Cumulative pair/step counts, not unique drones or collision incidents.
    UPROPERTY(BlueprintReadOnly, Category="Diagnostics") int64 SpacingViolationPairSteps = 0;
    UPROPERTY(BlueprintReadOnly, Category="Diagnostics") int64 OverlapPairSteps = 0;
    UPROPERTY(BlueprintReadOnly, Category="Diagnostics") int64 ObstacleEmergencyStops = 0;
    UPROPERTY(BlueprintReadOnly, Category="Diagnostics") int64 SpeedViolationSteps = 0;
    UPROPERTY(BlueprintReadOnly, Category="Diagnostics") int64 AccelerationViolationSteps = 0;
    UPROPERTY(BlueprintReadOnly, Category="Diagnostics") double PeakSpeedCmPerSecond = 0.0;
    UPROPERTY(BlueprintReadOnly, Category="Diagnostics") double PeakAccelerationCmPerSecondSquared = 0.0;
};


/** Work counters are separate from physical diagnostics and never affect simulation decisions. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FIstanaSwarmWorkCounters
{
    GENERATED_BODY()
    UPROPERTY(BlueprintReadOnly) int64 PathSearches = 0;
    UPROPERTY(BlueprintReadOnly) int64 ExpandedNodes = 0;
    UPROPERTY(BlueprintReadOnly) int64 CollisionQueries = 0;
    UPROPERTY(BlueprintReadOnly) int64 NeighborCandidates = 0;
    UPROPERTY(BlueprintReadOnly) int64 DiagnosticPairs = 0;
};
