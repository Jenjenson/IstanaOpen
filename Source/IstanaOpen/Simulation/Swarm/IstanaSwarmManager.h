#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "GameFramework/GameModeBase.h"
#include "Simulation/Swarm/IstanaSwarmSimulation.h"
#include "IstanaSwarmManager.generated.h"

class UStaticMeshComponent;
class ATargetPoint;

/** Presentation only. All motion and ground truth belongs to the solver. */
UCLASS()
class ISTANAOPEN_API AIstanaDroneVisual : public AActor
{
    GENERATED_BODY()
public:
    AIstanaDroneVisual();
    void ApplyState(const FIstanaDroneState& State, double FixedStepSeconds, double MaxTiltDegrees);
    void AnimateDisplayRotors(double PresentationSeconds);
private:
    UPROPERTY() TObjectPtr<UStaticMeshComponent> Body;
    UPROPERTY() TArray<TObjectPtr<UStaticMeshComponent>> DisplayRotors;
    FVector PreviousVelocityCmPerSecond = FVector::ZeroVector;
    double LastYawDegrees = 0.0;
    bool bHasPreviousState = false;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FIstanaSwarmStepped, int64, CompletedSteps);

/** Standalone harness or coordinator-driven adapter. Do not use both clocks at once. */
UCLASS(BlueprintType)
class ISTANAOPEN_API AIstanaSwarmManager : public AActor
{
    GENERATED_BODY()
public:
    AIstanaSwarmManager();
    virtual void Tick(float DeltaSeconds) override;
    virtual void Destroyed() override;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Swarm|Setup") TArray<FIstanaSwarmConfig> Swarms;
    // Reset/demo setup only. Shared InitializeSimulation inputs remain world-space.
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Swarm|Setup") bool bSpawnRelativeToManager = true;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Swarm|Navigation") bool bUseWorldCollision = true;
    // Triangle collision also supports imported scenery without simple collision hulls.
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Swarm|Navigation") bool bTraceComplexObstacles = true;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Swarm|Navigation") TEnumAsByte<ECollisionChannel> ObstacleTraceChannel = ECC_Visibility;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Swarm|Setup") FIstanaSwarmSettings Settings;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Swarm|Setup") TObjectPtr<UIstanaSwarmMovementPreset> MovementPreset;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Swarm|Setup") int32 Seed = 12345;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Swarm|Setup", meta=(ClampMin="0.001", ClampMax="0.1")) double FixedStepSeconds = 0.05;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Swarm|Setup") bool bAutoInitialize = true;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Swarm|Clock") bool bAutoAdvance = true;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Swarm|Clock", meta=(ClampMin="1", ClampMax="64")) int32 MaxStepsPerFrame = 8;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Swarm|Clock") bool bPaused = false;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Swarm|Presentation") bool bSpawnVisuals = true;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Swarm|Presentation") bool bDrawDebug = true;
    // Neighbor-radius spheres are useful in the standalone swarm harness but
    // can obscure the drone meshes in presentation-oriented integrations.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Swarm|Presentation") bool bDrawDroneNeighborRings = true;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Swarm|Demo") bool bEnableDemoKeyboard = false;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Swarm|Demo") TArray<FVector> DemoWaypointsCm;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Swarm|Demo") bool bLoopDemoRoute = true;

    // Explicit editor reference: accepts Target Point or BP_SwarmObjective. No name lookup.
    UPROPERTY(EditInstanceOnly, BlueprintReadWrite, Category="Swarm|Objective") TObjectPtr<ATargetPoint> ObjectiveTarget;
    // Takes precedence over DemoWaypoints when a marker is assigned.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Swarm|Objective") bool bFollowObjective = true;
    UPROPERTY(BlueprintReadOnly, Category="Swarm|Objective") FString ObjectiveStatus;

    UPROPERTY(BlueprintReadOnly, Category="Swarm|Diagnostics") double DroppedWallSeconds = 0.0;
    UPROPERTY(BlueprintAssignable, Category="Swarm") FIstanaSwarmStepped OnSimulationStepped;

    UFUNCTION(BlueprintCallable, Category="Istana|Swarm") virtual bool ResetSimulation(FString& Error);
    // Coordinator passes its own run ID and group configs; no policy observations are inferred here.
    UFUNCTION(BlueprintCallable, Category="Istana|Swarm") bool InitializeSimulation(const TArray<FIstanaSwarmConfig>& Configs,
        const FIstanaSwarmSettings& InSettings, int32 InSeed, double InFixedStepSeconds, FGuid InRunId, FString& Error);
    UFUNCTION(BlueprintCallable, Category="Istana|Swarm") bool SubmitCommand(const FIstanaSwarmCommand& Command, FString& Error);
    // Explicit single-step also works while paused. Turn off bAutoAdvance for coordinator ownership.
    UFUNCTION(BlueprintCallable, Category="Istana|Swarm") virtual void AdvanceOneStep();
    UFUNCTION(BlueprintPure, Category="Istana|Swarm") TArray<FIstanaDroneState> GetDroneStates() const;
    UFUNCTION(BlueprintPure, Category="Istana|Swarm") TArray<FIstanaSwarmGroupStatus> GetGroupStatuses() const;
    UFUNCTION(BlueprintPure, Category="Istana|Swarm") FIstanaSwarmDiagnostics GetDiagnostics() const;
    UFUNCTION(BlueprintPure, Category="Istana|Swarm") FGuid GetRunId() const;
    UFUNCTION(BlueprintPure, Category="Istana|Swarm") FIstanaSwarmWorkCounters GetWorkCounters() const;
    UFUNCTION(BlueprintCallable, Category="Istana|Swarm") void TogglePaused();
    UFUNCTION(BlueprintCallable, Category="Istana|Swarm") void ToggleDebug();
    UFUNCTION(BlueprintCallable, Category="Istana|Swarm") virtual void RestartDemo();

protected:
    bool PrepareSimulation(FIstanaSwarmSimulation& Candidate, const TArray<FIstanaSwarmConfig>& Configs,
        const FIstanaSwarmSettings& InSettings, int32 InSeed, double InStep, FGuid InRun, FString& Error);
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type Reason) override;
private:
    void RefreshVisuals();
    void ClearVisuals();
    void DrawDiagnostics() const;
    void UpdateObjective();
    bool CommandAllGroups(EIstanaSwarmCommandType Type, const FVector& Position, FString& Error);
    FIstanaSwarmSimulation Simulation;
    UPROPERTY(Transient) TArray<TObjectPtr<AIstanaDroneVisual>> Visuals;
    TArray<FString> DebugLabels;
    TArray<FColor> DebugColors;
    double AccumulatorSeconds = 0.0;
    TWeakObjectPtr<ATargetPoint> LastObjective;
    FVector LastObjectivePosition = FVector::ZeroVector;
    bool bHadObjective = false;
    bool bHasObjectiveAttempt = false;
};

/** Free-flight camera for the separate swarm demo; not used by the original map. */
UCLASS()
class ISTANAOPEN_API AIstanaSwarmDemoGameMode : public AGameModeBase
{
    GENERATED_BODY()
public:
    AIstanaSwarmDemoGameMode();
};
