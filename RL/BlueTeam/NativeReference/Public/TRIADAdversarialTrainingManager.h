#pragma once

#include "GameFramework/Actor.h"
#include "TRIADRLTrainingTypes.h"
#include "TRIADAdversarialTrainingManager.generated.h"

class ACesiumGeoreference;
class ATRIADDemoDroneActor;
class ATRIADSensorNodeActor;
class UTRIADProtectedZoneComponent;
class UTRIADSwarmControllerComponent;

/** Deterministic, simulation-only Red/Blue episode owner. */
UCLASS(BlueprintType, NotPlaceable)
class TRIADSENSORFUSION_API ATRIADAdversarialTrainingManager : public AActor
{
    GENERATED_BODY()

public:
    ATRIADAdversarialTrainingManager();

    static bool IsTrainingRequested();
    static bool IsEvaluationRequested();
    static bool LoadTrainingConfig(
        FTRIADRLTrainingConfig& OutConfig,
        FString& OutError,
        FString* OutConfigFingerprint = nullptr);

    UFUNCTION(BlueprintCallable, Category = "TRIAD|RL")
    bool ResetEpisode(int32 Seed, FTRIADRLStepResult& OutResult, FString& OutError);

    UFUNCTION(BlueprintCallable, Category = "TRIAD|RL")
    bool ApplyBlueAction(const FTRIADRLBlueAction& Action, FTRIADRLStepResult& OutResult, FString& OutError);

    UFUNCTION(BlueprintCallable, Category = "TRIAD|RL")
    bool ApplyRedDeploymentAction(
        const FTRIADRLRedDeploymentAction& Action,
        FTRIADRLStepResult& OutResult,
        FString& OutError);

    UFUNCTION(BlueprintCallable, Category = "TRIAD|RL")
    bool ApplyRedActions(const TArray<FTRIADRLRedAction>& Actions, FTRIADRLStepResult& OutResult, FString& OutError);

    UFUNCTION(BlueprintCallable, Category = "TRIAD|RL")
    bool GetEpisodeSnapshot(FTRIADRLStepResult& OutResult, FString& OutError) const;

    UFUNCTION(BlueprintCallable, Category = "TRIAD|RL")
    bool SetEvaluationLabels(const FString& BlueCheckpoint, const FString& RedCheckpoint,
        FTRIADRLStepResult& OutResult, FString& OutError);

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    bool bEvaluationOnly = false;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    TObjectPtr<UTRIADProtectedZoneComponent> ProtectedZone;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    FTRIADRLTrainingConfig TrainingConfig;

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void Tick(float DeltaSeconds) override;

private:
    void ClearEpisodeActors();
    bool ResolveMapObjective(FString& OutError);
    bool SpawnTargetsFromDeployment(const FTRIADRLRedDeploymentAction& Action, FString& OutError);
    bool ResolveSensorCatalogue(FString& OutError);
    bool CanPlaceCandidate(int32 Index) const;
    bool SpawnSensorFromCatalogue(int32 Index, const FVector2D& NormalizedPosition, FString& OutError);
    void BuildStepResult(FTRIADRLStepResult& OutResult) const;
    int32 EvaluateDetectedTargets();
    double ComputeMinimumZoneDistanceMeters() const;
    bool HasConstraintViolation() const;
    bool HasProtectedZoneEntry() const;
    ACesiumGeoreference* FindGeoreference() const;
    void DrawEvaluationOverlay();
    void LogTerminalEvaluation(const FTRIADRLStepResult& Result) const;

    UPROPERTY(Transient)
    TObjectPtr<ACesiumGeoreference> Georeference;

    UPROPERTY(Transient)
    TObjectPtr<AActor> ObjectiveActor;

    UPROPERTY(Transient)
    TArray<TObjectPtr<ATRIADSensorNodeActor>> SpawnedSensors;

    UPROPERTY(Transient)
    TArray<TObjectPtr<ATRIADDemoDroneActor>> SpawnedTargets;

    UPROPERTY(Transient)
    TArray<TObjectPtr<UTRIADSwarmControllerComponent>> TargetControllers;

    TArray<FTRIADRLTrackState> TargetTracks;
    TArray<int32> SensorDetectionCounts;
    /** Unique currently detected targets per catalogue profile (not per-sensor hit totals). */
    TArray<int32> CatalogueDetectionCounts;
    TArray<TArray<FVector>> TargetPaths;
    TArray<FTRIADRLMountObservation> ResolvedMounts;
    TArray<FVector> CandidateWorldPositions;
    TArray<int32> PlacedCatalogueIndices;
    ETRIADRLPhase Phase = ETRIADRLPhase::Inactive;
    ETRIADRLTerminationReason TerminationReason = ETRIADRLTerminationReason::None;
    int32 StepIndex = 0;
    int32 TransitionIndex = 0;
    int32 LastDetectedTargetCount = 0;
    int32 CurrentlyDetectedTargetCount = 0;
    int32 TrackedTargetCount = 0;
    int32 AllTargetsTrackedSteps = 0;
    int32 InvalidActionCount = 0;
    FTRIADRLRewardBreakdown LastStepRewards;
    FString LastBlueAction;
    FString LastRedDeploymentAction;
    TArray<FVector> LastRedMovementActions;
    FString BluePolicyLabel = TEXT("not loaded");
    FString RedPolicyLabel = TEXT("not loaded");
    FString ConfigFingerprint;
    double SpentSensorBudgetUnits = 0.0;
    double PreviousMinimumZoneDistanceMeters = 0.0;
    double AccumulatedBlueReward = 0.0;
    double AccumulatedRedReward = 0.0;
    int32 EpisodeSeed = 1;
};
