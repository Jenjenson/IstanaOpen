#pragma once

#include "CoreMinimal.h"
#include "Simulation/IstanaSimulationTypes.h"
#include "Simulation/Swarm/IstanaSwarmTypes.h"

/** Value-only solver. An optional collision query supplies level geometry without owning actors. */
class ISTANAOPEN_API FIstanaSwarmSimulation
{
public:
    using FCollisionBatch = TFunction<void(bool)>;
    using FCollisionQuery = TFunction<bool(const FVector&, const FVector&, double)>;
    // Transactional: invalid input leaves an already running simulation unchanged.
    bool Initialize(const TArray<FIstanaSwarmConfig>& Configs, const FIstanaSwarmSettings& Settings,
        int32 Seed, double FixedStepSeconds, const FGuid& RunId, FString& Error,
        FCollisionQuery CollisionQuery = FCollisionQuery(), FCollisionBatch CollisionBatch = FCollisionBatch());
    bool SubmitCommand(const FIstanaSwarmCommand& Command, FString& Error);
    void Step();
    bool IsInitialized() const { return bInitialized; }
    const TArray<FIstanaDroneState>& GetStates() const { return States; }
    const FIstanaSwarmDiagnostics& GetDiagnostics() const { return Diagnostics; }
    const FIstanaSwarmSettings& GetSettings() const { return Settings; }
    const FIstanaSwarmWorkCounters& GetWorkCounters() const { return Work; }
    FGuid GetRunId() const { return RunId; }
    double GetFixedStepSeconds() const { return FixedStepSeconds; }
    TArray<FIstanaSwarmGroupStatus> GetGroupStatuses() const;
#if WITH_DEV_AUTOMATION_TESTS
    FString GetNavigationFingerprint() const;
#endif
    int64 GetNextCommandSequence(int32 GroupId) const;

private:
    struct FGroup
    {
        int32 Id = INDEX_NONE;
        TArray<int32> Members;
        FVector HoldTarget = FVector::ZeroVector;
        TArray<FVector> Waypoints;
        int32 WaypointIndex = 0;
        bool bLoop = false;
        bool bAllowPartialPath = false;
        bool bRouteCompleted = false;
        EIstanaSwarmCommandType Mode = EIstanaSwarmCommandType::Hold;
        double CruiseSpeed = 0;
        double Spacing = 0;
        int64 LastSequence = INDEX_NONE;
    };
    FVector Centroid(int32 GroupId) const;
    FVector Target(const FGroup& Group) const;
    bool IsFreePosition(const FVector& Position) const;
    bool SegmentHitsObstacle(const FVector& Start, const FVector& End) const;
    bool IsSegmentFree(const FVector& Start, const FVector& End, double Padding) const;
    bool FindPath(const FVector& Start, const FVector& End, TArray<FVector>& Path, bool bAllowPartial = false) const;
    struct FNavigationRoute
    {
        TArray<FVector> Points;
        FVector Goal = FVector::ZeroVector;
        int32 NextPoint = 0;
        int64 RetryStep = 0;
        bool bBlocked = false;
        bool bPartialPath = false;
    };
    TArray<FNavigationRoute> NavigationRoutes;
    FCollisionQuery CollisionQuery;
    FCollisionBatch CollisionBatch;
    TArray<FIstanaDroneState> NextStates;
    // Previous-step commanded accelerations make jerk limiting deterministic.
    TArray<FVector> Accelerations;
    TArray<FVector> NextAccelerations;
    TArray<int32> StateGroups;
    FIstanaSwarmSettings Settings;
    double FixedStepSeconds = 0.05;
    FGuid RunId;
    bool bInitialized = false;
    TArray<FIstanaDroneState> States;
    TArray<FVector> FormationOffsets;
    TArray<FGroup> Groups;
    FIstanaSwarmDiagnostics Diagnostics;
    mutable FIstanaSwarmWorkCounters Work;
};
