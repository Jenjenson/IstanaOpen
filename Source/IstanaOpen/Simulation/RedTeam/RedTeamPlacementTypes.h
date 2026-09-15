#pragma once
#include "CoreMinimal.h"
#include "Simulation/IstanaSimulationTypes.h"
#include "Simulation/Swarm/IstanaSwarmTypes.h"
#include "RedTeamPlacementTypes.generated.h"

UENUM(BlueprintType)
enum class ERedTeamPlacementSource : uint8 { SeededLayout, AgentPlacement };
UENUM(BlueprintType)
enum class ERedTeamEpisodePhase : uint8 { Idle, AwaitingPlacement, Running, Completed, Cancelled };

USTRUCT(BlueprintType)
struct ISTANAOPEN_API FRedTeamPlacementCenter
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite) int32 GroupId = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) FVector CenterWorldCm = FVector::ZeroVector;
};

/** Immutable environment configuration advertised before the policy chooses centers. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FRedTeamPlacementContext
{
    GENERATED_BODY()
    UPROPERTY(BlueprintReadOnly) int32 SchemaVersion = 1;
    UPROPERTY(BlueprintReadOnly) FGuid RunId;
    UPROPERTY(BlueprintReadOnly) int64 Revision = 0;
    UPROPERTY(BlueprintReadOnly) int64 WorldRevision = 0;
    UPROPERTY(BlueprintReadOnly) FString ObjectiveId;
    UPROPERTY(BlueprintReadOnly) FVector ObjectiveWorldCm = FVector::ZeroVector;
    UPROPERTY(BlueprintReadOnly) int32 MemberSeed = 0;
    UPROPERTY(BlueprintReadOnly) int32 GroupCount = 0;
    UPROPERTY(BlueprintReadOnly) int32 MembersPerGroup = 0;
    UPROPERTY(BlueprintReadOnly) double SpreadRadiusCm = 0;
    UPROPERTY(BlueprintReadOnly) double MinRadiusCm = 0;
    UPROPERTY(BlueprintReadOnly) double MaxRadiusCm = 0;
    UPROPERTY(BlueprintReadOnly) double HeightOffsetCm = 0;
    UPROPERTY(BlueprintReadOnly) double FixedStepSeconds = 0;
    UPROPERTY(BlueprintReadOnly) FIstanaSwarmSettings Movement;
    UPROPERTY(BlueprintReadOnly) bool bWorldCollision = true;
    UPROPERTY(BlueprintReadOnly) bool bComplexCollision = true;
    UPROPERTY(BlueprintReadOnly) int32 CollisionChannel = 0;
};

USTRUCT(BlueprintType)
struct ISTANAOPEN_API FRedTeamPlacementAction
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite) int32 SchemaVersion = 1;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) FGuid RunId;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) int64 Revision = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) int64 RequestId = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) TArray<FRedTeamPlacementCenter> Centers;
};

USTRUCT(BlueprintType)
struct ISTANAOPEN_API FRedTeamPlacementResult
{
    GENERATED_BODY()
    UPROPERTY(BlueprintReadOnly) bool bAccepted = false;
    UPROPERTY(BlueprintReadOnly) FGuid RunId;
    UPROPERTY(BlueprintReadOnly) int64 RequestId = 0;
    UPROPERTY(BlueprintReadOnly) FString Error;
    UPROPERTY(BlueprintReadOnly) TArray<FIstanaSwarmConfig> AcceptedGroups;
    UPROPERTY(BlueprintReadOnly) TArray<FIstanaDroneState> InitialStates;
};

/** Own-team state only. No evaluator truth or future sensor observations are inferred. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FRedTeamEpisodeObservation
{
    GENERATED_BODY()
    UPROPERTY(BlueprintReadOnly) FGuid RunId;
    UPROPERTY(BlueprintReadOnly) ERedTeamEpisodePhase Phase = ERedTeamEpisodePhase::Idle;
    UPROPERTY(BlueprintReadOnly) int64 CompletedSteps = 0;
    UPROPERTY(BlueprintReadOnly) TArray<FIstanaDroneState> Drones;
    UPROPERTY(BlueprintReadOnly) TArray<FIstanaSwarmGroupStatus> Groups;
    UPROPERTY(BlueprintReadOnly) FIstanaSwarmDiagnostics Diagnostics;
    UPROPERTY(BlueprintReadOnly) bool bTerminated = false;
    UPROPERTY(BlueprintReadOnly) bool bTruncated = false;
    UPROPERTY(BlueprintReadOnly) FString Reason;
    UPROPERTY(BlueprintReadOnly) bool bHasReward = false;
    UPROPERTY(BlueprintReadOnly) double Reward = 0;
};
