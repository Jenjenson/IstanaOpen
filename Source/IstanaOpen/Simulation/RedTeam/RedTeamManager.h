#pragma once

#include "CoreMinimal.h"
#include "Simulation/Swarm/IstanaSwarmManager.h"
#include "Simulation/RedTeam/RedTeamPlacementPolicy.h"
#include "RedTeamManager.generated.h"

/** Placeable multi-swarm spawner. All groups share ObjectiveTarget and one solver/clock. */
UCLASS(BlueprintType, Blueprintable, meta=(DisplayName="Red Team Manager"))
class ISTANAOPEN_API ARedTeamManager : public AIstanaSwarmManager
{
    GENERATED_BODY()
public:
    ARedTeamManager();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Red Team|Spawning", meta=(ClampMin="1", ClampMax="256"))
    int32 NumberOfSwarms = 3;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Red Team|Spawning", meta=(ClampMin="1", ClampMax="256"))
    int32 DronesPerSwarm = 12;
    // Horizontal distance of each spawn center from the objective, in centimeters.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Red Team|Spawning", meta=(ClampMin="0"))
    double MinSpawnRadiusCm = 3000.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Red Team|Spawning", meta=(ClampMin="0"))
    double MaxSpawnRadiusCm = 6000.0;
    // Individual drones are sampled within a sphere around each generated center.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Red Team|Spawning", meta=(ClampMin="0"))
    double SwarmSpreadRadiusCm = 400.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Red Team|Spawning")
    double SpawnHeightOffsetCm = 0.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Red Team|Spawning", meta=(ClampMin="1", ClampMax="128"))
    int32 MaxSpawnLayoutAttempts = 16;
    UPROPERTY(VisibleInstanceOnly, BlueprintReadOnly, Category="Red Team|Status")
    TArray<FIstanaSwarmConfig> SpawnedGroups;
    UPROPERTY(VisibleInstanceOnly, BlueprintReadOnly, Category="Red Team|Status")
    FString SpawnStatus;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Red Team|Agent")
    ERedTeamPlacementSource PlacementSource = ERedTeamPlacementSource::SeededLayout;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Red Team|Agent")
    TScriptInterface<IRedTeamPlacementPolicy> PlacementPolicy;
    UPROPERTY(VisibleInstanceOnly, BlueprintReadOnly, Category="Red Team|Agent")
    ERedTeamEpisodePhase EpisodePhase = ERedTeamEpisodePhase::Idle;
    // The environment increments this when streamed geometry or placement constraints change.
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Red Team|Agent") int64 PlacementWorldRevision = 0;
    UFUNCTION(BlueprintCallable, Category="Red Team|Agent")
    bool BeginPlacementEpisode(int32 EpisodeSeed, FRedTeamPlacementContext& Context, FString& Error);
    UFUNCTION(BlueprintCallable, Category="Red Team|Agent")
    bool ValidatePlacement(const FRedTeamPlacementAction& Action, FString& Error);
    UFUNCTION(BlueprintCallable, Category="Red Team|Agent")
    FRedTeamPlacementResult SubmitPlacement(const FRedTeamPlacementAction& Action);
    UFUNCTION(BlueprintCallable, Category="Red Team|Agent")
    bool AdvanceEpisode(int32 Steps, FRedTeamEpisodeObservation& Observation, FString& Error);
    UFUNCTION(BlueprintPure, Category="Red Team|Agent")
    FRedTeamPlacementContext GetPlacementContext() const { return PlacementContext; }
    UFUNCTION(BlueprintPure, Category="Red Team|Agent")
    FRedTeamEpisodeObservation GetEpisodeObservation() const;
    UFUNCTION(BlueprintCallable, Category="Red Team|Agent") void CancelEpisode(const FString& Reason);
    UFUNCTION(BlueprintCallable, Category="Red Team|Agent") void NotifyPlacementWorldChanged();
    // Optional evaluator owns reward semantics; no reward is fabricated by the manager.
    UFUNCTION(BlueprintNativeEvent, Category="Red Team|Agent")
    void EvaluateEpisode(FRedTeamEpisodeObservation& Observation);
    virtual void AdvanceOneStep() override;
    virtual void RestartDemo() override;
    virtual void Tick(float DeltaSeconds) override;

    virtual bool ResetSimulation(FString& Error) override;
    UFUNCTION(BlueprintCallable, Category="Red Team")
    bool SpawnSwarms(FString& Error) { return ResetSimulation(Error); }
protected:
    virtual void BeginPlay() override;
private:
    bool BuildPlacement(const FRedTeamPlacementAction& Action, TArray<FIstanaSwarmConfig>& Groups, FString& Error) const;
    FRedTeamPlacementContext PlacementContext;
    TMap<int64, FRedTeamPlacementAction> RequestActions;
    TMap<int64, FRedTeamPlacementResult> RequestResults;
    int64 LastRequestId = -1;
    bool bEpisodeStep = false;
    FString EpisodeReason;
};
