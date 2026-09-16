#include "Simulation/RedTeam/RedTeamManager.h"
#include "Simulation/BlueTeam/BlueTeamCoordinator.h"
#include "Engine/TargetPoint.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"

ARedTeamManager::ARedTeamManager()
{
    // Generated centers are world coordinates around the objective, independent of this actor.
    bSpawnRelativeToManager = false;
    Swarms.Reset();
}

bool ARedTeamManager::ResetSimulation(FString& Error)
{
    if (PlacementSource == ERedTeamPlacementSource::AgentPlacement)
    { FRedTeamPlacementContext Context; return BeginPlacementEpisode(Seed, Context, Error); }
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


void ARedTeamManager::BeginPlay()
{
    // Explicit profiling-only overrides never modify saved level/preset assets.
    if (FParse::Param(FCommandLine::Get(), TEXT("IstanaSwarmProfile")))
    {
        PlacementSource = ERedTeamPlacementSource::SeededLayout;
        int32 Count = NumberOfSwarms;
        FParse::Value(FCommandLine::Get(), TEXT("SwarmProfileCount="), Count);
        if (Count == 0) { bAutoInitialize = false; bAutoAdvance = false; bDrawDebug = false; }
        else
        {
            bAutoInitialize = true; bAutoAdvance = true; bPaused = false;
            NumberOfSwarms = FMath::Clamp(Count, 1, 21); DronesPerSwarm = 12;
            Settings = MovementPreset ? MovementPreset->Settings : Settings; MovementPreset = nullptr;
            Settings.MaxDrones = 256;
        }
        bDrawDebug = Count > 0 && !FParse::Param(FCommandLine::Get(), TEXT("SwarmProfileNoDebug"));
        bSpawnVisuals = Count > 0 && !FParse::Param(FCommandLine::Get(), TEXT("SwarmProfileNoVisuals"));
    }
    if (PlacementSource == ERedTeamPlacementSource::AgentPlacement)
    {
        bAutoInitialize = false; bAutoAdvance = false; bEnableDemoKeyboard = false;
        Super::BeginPlay();
        FRedTeamPlacementContext Context; FString Error;
        if (!BeginPlacementEpisode(Seed, Context, Error)) SpawnStatus = Error;
        return;
    }
    Super::BeginPlay();
}

void ARedTeamManager::Tick(float DeltaSeconds)
{
    if (PlacementSource == ERedTeamPlacementSource::AgentPlacement) bAutoAdvance = false;
    Super::Tick(DeltaSeconds);
}

void ARedTeamManager::AdvanceOneStep()
{
    if (PlacementSource == ERedTeamPlacementSource::AgentPlacement) return;
    Super::AdvanceOneStep();
}

bool ARedTeamManager::BeginPlacementEpisode(int32 EpisodeSeed, FRedTeamPlacementContext& Context, FString& Error)
{
    Error.Reset();
    if (bEpisodeStep) { Error = TEXT("Cannot reset during a step callback."); return false; }
    if (PlacementSource != ERedTeamPlacementSource::AgentPlacement)
    { Error = TEXT("PlacementSource must be AgentPlacement."); return false; }
    if (!IsValid(ObjectiveTarget) || ObjectiveTarget->GetActorLocation().ContainsNaN())
    { Error = TEXT("ObjectiveTarget must be valid and finite."); return false; }
    const auto& Movement = MovementPreset ? MovementPreset->Settings : Settings;
    if (NumberOfSwarms < 1 || NumberOfSwarms > 256 || DronesPerSwarm < 1 || DronesPerSwarm > 256
        || int64(NumberOfSwarms) * DronesPerSwarm > Movement.MaxDrones
        || !FMath::IsFinite(MinSpawnRadiusCm) || !FMath::IsFinite(MaxSpawnRadiusCm)
        || MinSpawnRadiusCm < 0 || MaxSpawnRadiusCm < MinSpawnRadiusCm
        || !FMath::IsFinite(SwarmSpreadRadiusCm) || SwarmSpreadRadiusCm < 0
        || (DronesPerSwarm > 1 && SwarmSpreadRadiusCm == 0) || !FMath::IsFinite(SpawnHeightOffsetCm))
    { Error = TEXT("Invalid environment population or placement constraints."); return false; }
    // Validate movement/clock/world availability before advertising a policy context.
    FIstanaSwarmSimulation ConfigurationProbe;
    if (!PrepareSimulation(ConfigurationProbe, {}, Movement, EpisodeSeed, FixedStepSeconds, FGuid::NewGuid(), Error)) return false;
    if (IsValid(BlueCoordinator) && !BlueCoordinator->ValidateConfiguration(FixedStepSeconds, Error)) return false;
    bAutoAdvance = false; bEnableDemoKeyboard = false;
    PlacementContext.RunId = FGuid::NewGuid(); ++PlacementContext.Revision;
    PlacementContext.WorldRevision = PlacementWorldRevision;
    PlacementContext.ObjectiveId = ObjectiveTarget->GetPathName();
    PlacementContext.ObjectiveWorldCm = ObjectiveTarget->GetActorLocation();
    PlacementContext.MemberSeed = EpisodeSeed;
    PlacementContext.GroupCount = NumberOfSwarms; PlacementContext.MembersPerGroup = DronesPerSwarm;
    PlacementContext.SpreadRadiusCm = SwarmSpreadRadiusCm;
    PlacementContext.MinRadiusCm = MinSpawnRadiusCm; PlacementContext.MaxRadiusCm = MaxSpawnRadiusCm;
    PlacementContext.HeightOffsetCm = SpawnHeightOffsetCm;
    PlacementContext.FixedStepSeconds = FixedStepSeconds; PlacementContext.Movement = Movement;
    PlacementContext.bWorldCollision = bUseWorldCollision;
    PlacementContext.bComplexCollision = bTraceComplexObstacles;
    PlacementContext.CollisionChannel = int32(ObstacleTraceChannel.GetValue());
    LastRequestId = -1; RequestActions.Reset(); RequestResults.Reset(); EpisodeReason.Reset();
    EpisodePhase = ERedTeamEpisodePhase::AwaitingPlacement;
    SpawnStatus = TEXT("Awaiting agent placement; previous simulation is frozen.");
    Context = PlacementContext;
    if (IsValid(BlueCoordinator)) BlueCoordinator->BeginEpisode(Context);
    if (UObject* Policy = PlacementPolicy.GetObject())
    {
        IRedTeamPlacementPolicy::Execute_ResetPlacementPolicy(Policy, Context);
        IRedTeamPlacementPolicy::Execute_RequestPlacement(Policy, Context);
    }
    return true;
}

bool ARedTeamManager::BuildPlacement(const FRedTeamPlacementAction& Action,
    TArray<FIstanaSwarmConfig>& Groups, FString& Error) const
{
    Error.Reset();
    auto Fail = [&](const TCHAR* Reason) { Error = Reason; return false; };
    if (PlacementSource != ERedTeamPlacementSource::AgentPlacement || EpisodePhase != ERedTeamEpisodePhase::AwaitingPlacement)
        return Fail(TEXT("phase: episode is not awaiting placement."));
    if (Action.SchemaVersion != 1) return Fail(TEXT("schemaVersion: expected 1."));
    if (Action.RunId != PlacementContext.RunId || Action.Revision != PlacementContext.Revision)
        return Fail(TEXT("runId/revision: stale placement context."));
    if (Action.RequestId < 0 || Action.RequestId <= LastRequestId)
        return Fail(TEXT("requestId: must increase after a rejected request."));
    if (!IsValid(ObjectiveTarget) || ObjectiveTarget->GetPathName() != PlacementContext.ObjectiveId
        || ObjectiveTarget->GetActorLocation() != PlacementContext.ObjectiveWorldCm
        || PlacementWorldRevision != PlacementContext.WorldRevision
        || bUseWorldCollision != PlacementContext.bWorldCollision || bTraceComplexObstacles != PlacementContext.bComplexCollision
        || int32(ObstacleTraceChannel.GetValue()) != PlacementContext.CollisionChannel)
        return Fail(TEXT("context: objective or world changed; begin a new placement episode."));
    if (RequestActions.Num() >= 256) return Fail(TEXT("request budget: begin a new episode after 256 placement requests."));
    if (Action.Centers.Num() != PlacementContext.GroupCount)
        return Fail(TEXT("centers: supply exactly one center for every group ID [0, groupCount)."));
    TSet<int32> Seen;
    for (const auto& Center : Action.Centers)
    {
        if (Center.GroupId < 0 || Center.GroupId >= PlacementContext.GroupCount || Seen.Contains(Center.GroupId))
            return Fail(TEXT("centers.groupId: duplicate or unknown group."));
        Seen.Add(Center.GroupId);
        if (Center.CenterWorldCm.ContainsNaN()) return Fail(TEXT("centers.centerWorldCm: coordinates must be finite."));
        const double Radius = FVector::Dist2D(Center.CenterWorldCm, PlacementContext.ObjectiveWorldCm);
        if (Radius < PlacementContext.MinRadiusCm || Radius > PlacementContext.MaxRadiusCm
            || FMath::Abs(Center.CenterWorldCm.Z - (PlacementContext.ObjectiveWorldCm.Z + PlacementContext.HeightOffsetCm)) > 0.001)
            return Fail(TEXT("centers.centerWorldCm: outside advertised radius/height constraints."));
        for (const auto& Group : Groups)
            if (FVector::Dist(Group.SpawnOriginCm, Center.CenterWorldCm) < 2 * PlacementContext.SpreadRadiusCm + PlacementContext.Movement.SpacingCm)
                return Fail(TEXT("centers: spawn regions overlap."));
        FIstanaSwarmConfig Group;
        Group.GroupId = Center.GroupId; Group.DroneCount = PlacementContext.MembersPerGroup;
        Group.SpawnOriginCm = Center.CenterWorldCm; Group.SpawnRadiusCm = PlacementContext.SpreadRadiusCm;
        Group.MovementPresetId = TEXT("Boids"); Groups.Add(Group);
    }
    Groups.Sort([](const auto& A, const auto& B) { return A.GroupId < B.GroupId; });
    return true;
}

bool ARedTeamManager::ValidatePlacement(const FRedTeamPlacementAction& Action, FString& Error)
{
    TArray<FIstanaSwarmConfig> Groups;
    if (!BuildPlacement(Action, Groups, Error)) return false;
    FIstanaSwarmSimulation Candidate;
    return PrepareSimulation(Candidate, Groups, PlacementContext.Movement, PlacementContext.MemberSeed,
        PlacementContext.FixedStepSeconds, PlacementContext.RunId, Error);
}

FRedTeamPlacementResult ARedTeamManager::SubmitPlacement(const FRedTeamPlacementAction& Action)
{
    FRedTeamPlacementResult Result; Result.RunId = Action.RunId; Result.RequestId = Action.RequestId;
    const FRedTeamPlacementAction* Previous = Action.RunId == PlacementContext.RunId ? RequestActions.Find(Action.RequestId) : nullptr;
    if (Previous)
    {
        bool bSame = Action.SchemaVersion == Previous->SchemaVersion && Action.Revision == Previous->Revision
            && Action.Centers.Num() == Previous->Centers.Num();
        for (int32 I = 0; bSame && I < Action.Centers.Num(); ++I)
            bSame = Action.Centers[I].GroupId == Previous->Centers[I].GroupId
                && Action.Centers[I].CenterWorldCm == Previous->Centers[I].CenterWorldCm;
        if (bSame) return RequestResults[Action.RequestId];
        Result.Error = TEXT("requestId: reused with a different payload."); return Result;
    }
    TArray<FIstanaSwarmConfig> Groups;
    if (BuildPlacement(Action, Groups, Result.Error)
        && InitializeSimulation(Groups, PlacementContext.Movement, PlacementContext.MemberSeed,
            PlacementContext.FixedStepSeconds, PlacementContext.RunId, Result.Error))
    {
        SpawnedGroups = Groups; Result.AcceptedGroups = Groups; Result.InitialStates = GetDroneStates();
        Result.bAccepted = true; bFollowObjective = true; bAutoAdvance = false;
        EpisodePhase = ERedTeamEpisodePhase::Running;
        if (IsValid(BlueCoordinator)) BlueCoordinator->CaptureInitialStates(Result.InitialStates);
        SpawnStatus = TEXT("Agent placement accepted; explicit fixed-step clock.");
    }
    // Invalid/stale context traffic cannot poison the current episode's request sequence.
    if (Action.RunId == PlacementContext.RunId && Action.Revision == PlacementContext.Revision && Action.RequestId > LastRequestId && RequestActions.Num() < 256)
    { LastRequestId = Action.RequestId; RequestActions.Add(Action.RequestId, Action); RequestResults.Add(Action.RequestId, Result); }
    UE_LOG(LogTemp, Display, TEXT("RedTeam placement run=%s request=%lld accepted=%d %s"),
        *Action.RunId.ToString(), Action.RequestId, Result.bAccepted, *Result.Error);
    return Result;
}

FRedTeamEpisodeObservation ARedTeamManager::GetEpisodeObservation() const
{
    FRedTeamEpisodeObservation Observation;
    Observation.RunId = PlacementContext.RunId; Observation.Phase = EpisodePhase; Observation.Reason = EpisodeReason;
    Observation.bTerminated = EpisodePhase == ERedTeamEpisodePhase::Completed;
    Observation.bTruncated = EpisodePhase == ERedTeamEpisodePhase::Cancelled;
    // Never expose a retained previous run as the new episode's observation.
    if (GetRunId() == PlacementContext.RunId)
    {
        Observation.Drones = GetDroneStates(); Observation.Groups = GetGroupStatuses();
        Observation.Diagnostics = GetDiagnostics(); Observation.CompletedSteps = Observation.Diagnostics.ExecutedSteps;
    }
    return Observation;
}

bool ARedTeamManager::AdvanceEpisode(int32 Steps, FRedTeamEpisodeObservation& Observation, FString& Error)
{
    Error.Reset();
    if (PlacementSource != ERedTeamPlacementSource::AgentPlacement || EpisodePhase != ERedTeamEpisodePhase::Running
        || GetRunId() != PlacementContext.RunId || Steps < 1 || Steps > 1000 || bEpisodeStep)
    { Error = TEXT("step: requires a running episode and 1..1000 fixed steps, without reentrant stepping."); Observation = GetEpisodeObservation(); return false; }
    if (IsValid(BlueCoordinator) && !BlueCoordinator->CanAdvance(Error))
    { Observation = GetEpisodeObservation(); return false; }
    for (int32 I = 0; I < Steps && EpisodePhase == ERedTeamEpisodePhase::Running; ++I)
    {
        bEpisodeStep = true; Super::AdvanceOneStep();
        Observation = GetEpisodeObservation(); EvaluateEpisode(Observation);
        if (IsValid(BlueCoordinator)) BlueCoordinator->Evaluate(Observation);
        bEpisodeStep = false;
        if (EpisodePhase == ERedTeamEpisodePhase::Cancelled)
        { Observation.bTruncated = true; Observation.Reason = EpisodeReason; }
        if (Observation.bTerminated || Observation.bTruncated)
        {
            EpisodePhase = Observation.bTerminated ? ERedTeamEpisodePhase::Completed : ERedTeamEpisodePhase::Cancelled;
            EpisodeReason = Observation.Reason; Observation.Phase = EpisodePhase;
        }
    }
    return true;
}

void ARedTeamManager::EvaluateEpisode_Implementation(FRedTeamEpisodeObservation& Observation) {}
void ARedTeamManager::CancelEpisode(const FString& Reason)
{
    EpisodePhase = ERedTeamEpisodePhase::Cancelled; EpisodeReason = Reason; bAutoAdvance = false;
}
void ARedTeamManager::NotifyPlacementWorldChanged() { ++PlacementWorldRevision; }

void ARedTeamManager::RestartDemo()
{
    if (PlacementSource == ERedTeamPlacementSource::AgentPlacement)
    { FRedTeamPlacementContext Context; FString Error; if (!BeginPlacementEpisode(Seed, Context, Error)) SpawnStatus = Error; }
    else Super::RestartDemo();
}
