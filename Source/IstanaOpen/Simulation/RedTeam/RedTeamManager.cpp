#include "Simulation/RedTeam/RedTeamManager.h"
#include "Engine/TargetPoint.h"

ARedTeamManager::ARedTeamManager()
{
    // Generated centers are world coordinates around the objective, independent of this actor.
    bSpawnRelativeToManager = false;
    Swarms.Reset();
}

bool ARedTeamManager::ResetSimulation(FString& Error)
{
    auto Fail = [&](const FString& Reason)
    {
        Error = Reason;
        SpawnStatus = Reason;
        return false;
    };
    if (!IsValid(ObjectiveTarget)) return Fail(TEXT("Assign Objective Target before spawning swarms."));
    const FIstanaSwarmSettings& Movement = MovementPreset ? MovementPreset->Settings : Settings;
    if (NumberOfSwarms < 1 || NumberOfSwarms > 256 || DronesPerSwarm < 1 || DronesPerSwarm > 256
        || int64(NumberOfSwarms) * DronesPerSwarm > Movement.MaxDrones)
        return Fail(TEXT("Swarm count x drones per swarm must fit the movement settings' Max Drones (up to 256)."));
    if (!FMath::IsFinite(MinSpawnRadiusCm) || !FMath::IsFinite(MaxSpawnRadiusCm)
        || MinSpawnRadiusCm < 0 || MaxSpawnRadiusCm < MinSpawnRadiusCm
        || !FMath::IsFinite(SwarmSpreadRadiusCm) || SwarmSpreadRadiusCm < 0
        || !FMath::IsFinite(SpawnHeightOffsetCm) || MaxSpawnLayoutAttempts < 1 || MaxSpawnLayoutAttempts > 128
        || (DronesPerSwarm > 1 && SwarmSpreadRadiusCm <= 0))
        return Fail(TEXT("Check spawn radii, height and layout attempts. Multiple drones require a positive spread radius."));

    FRandomStream Random(Seed);
    const FVector ObjectivePosition = ObjectiveTarget->GetActorLocation();
    for (int32 Attempt = 0; Attempt < MaxSpawnLayoutAttempts; ++Attempt)
    {
        TArray<FIstanaSwarmConfig> Proposed;
        const double Rotation = Random.FRand() * 2.0 * PI;
        bool bLayoutFits = true;
        for (int32 GroupIndex = 0; GroupIndex < NumberOfSwarms; ++GroupIndex)
        {
            // One angular sector per group distributes groups around the target.
            const double Angle = Rotation + (GroupIndex + Random.FRand()) * 2.0 * PI / NumberOfSwarms;
            const double Radius = FMath::Lerp(MinSpawnRadiusCm, MaxSpawnRadiusCm, double(Random.FRand()));
            FIstanaSwarmConfig Group;
            Group.GroupId = GroupIndex;
            Group.DroneCount = DronesPerSwarm;
            Group.SpawnOriginCm = ObjectivePosition + FVector(FMath::Cos(Angle) * Radius,
                FMath::Sin(Angle) * Radius, SpawnHeightOffsetCm);
            Group.SpawnRadiusCm = SwarmSpreadRadiusCm;
            Group.MovementPresetId = TEXT("Boids");
            for (const FIstanaSwarmConfig& Existing : Proposed)
                if (FVector::Dist(Group.SpawnOriginCm, Existing.SpawnOriginCm) < 2 * SwarmSpreadRadiusCm + Movement.SpacingCm)
                { bLayoutFits = false; break; }
            if (!bLayoutFits) break;
            Proposed.Add(Group);
        }
        if (!bLayoutFits) { Error = TEXT("Spawn regions overlap; increase the distance from the objective or reduce spread/count."); continue; }
        // Initializer validates collision and spacing. Failure preserves the current simulation.
        if (!InitializeSimulation(Proposed, Movement, Seed, FixedStepSeconds, FGuid::NewGuid(), Error)) continue;
        SpawnedGroups = MoveTemp(Proposed);
        bFollowObjective = true;
        SpawnStatus = FString::Printf(TEXT("Spawned %d swarms / %d drones around the shared objective."),
            NumberOfSwarms, NumberOfSwarms * DronesPerSwarm);
        Error.Reset();
        return true;
    }
    return Fail(FString::Printf(TEXT("Could not place swarms after %d layouts. %s"), MaxSpawnLayoutAttempts, *Error));
}
