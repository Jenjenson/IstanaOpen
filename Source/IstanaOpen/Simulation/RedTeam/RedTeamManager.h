#pragma once

#include "CoreMinimal.h"
#include "Simulation/Swarm/IstanaSwarmManager.h"
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

    virtual bool ResetSimulation(FString& Error) override;
    UFUNCTION(BlueprintCallable, Category="Red Team")
    bool SpawnSwarms(FString& Error) { return ResetSimulation(Error); }
};
