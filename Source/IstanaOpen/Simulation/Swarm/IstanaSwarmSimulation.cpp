#include "Simulation/Swarm/IstanaSwarmSimulation.h"
#include <queue>

namespace
{
    bool Finite(const FVector& V)
    {
        return FMath::IsFinite(V.X) && FMath::IsFinite(V.Y) && FMath::IsFinite(V.Z);
    }
    bool Positive(double V) { return FMath::IsFinite(V) && V > 0; }
    bool Nonnegative(double V) { return FMath::IsFinite(V) && V >= 0; }

    FVector LimitTurn(const FVector& Previous, const FVector& Next, double MaxRadians)
    {
        if (Previous.IsNearlyZero() || Next.IsNearlyZero()) return Next;
        const FVector From = Previous.GetSafeNormal();
        const FVector To = Next.GetSafeNormal();
        const double Angle = FMath::Acos(FMath::Clamp(FVector::DotProduct(From, To), -1.0, 1.0));
        if (Angle <= MaxRadians) return Next;
        FVector Axis = FVector::CrossProduct(From, To).GetSafeNormal();
        if (Axis.IsNearlyZero())
        {
            // Deterministic axis for a 180-degree turn.
            Axis = FVector::CrossProduct(From, FVector::UpVector).GetSafeNormal();
            if (Axis.IsNearlyZero()) Axis = FVector::RightVector;
        }
        return FQuat(Axis, MaxRadians).RotateVector(From) * Next.Size();
    }
}

bool FIstanaSwarmSimulation::IsFreePosition(const FVector& P) const
{
    return IsSegmentFree(P, P, 0);
}

bool FIstanaSwarmSimulation::SegmentHitsObstacle(const FVector& Start, const FVector& End) const
{
    return !IsSegmentFree(Start, End, 0);
}

bool FIstanaSwarmSimulation::IsSegmentFree(const FVector& Start, const FVector& End, double Padding) const
{
    if (!Finite(Start) || !Finite(End)) return false;
    const double Radius = Settings.DroneRadiusCm + Padding;
    return !CollisionQuery || !CollisionQuery(Start, End, Radius);
}

bool FIstanaSwarmSimulation::FindPath(const FVector& Start, const FVector& End, TArray<FVector>& Path, bool bAllowPartial) const
{
    Path.Reset();
    if (!Finite(End) || !IsFreePosition(Start)) return false;
    const bool bEndFree = IsFreePosition(End);
    if (!bAllowPartial && !bEndFree) return false;
    if (bEndFree && Start.Equals(End, UE_SMALL_NUMBER)) { Path.Add(End); return true; }
    const double Padding = Settings.NavigationClearanceCm;
    if (bEndFree && IsSegmentFree(Start, End, Padding)) { Path.Add(End); return true; }
    // Endpoint clearance may be smaller than cruising clearance (e.g. a spawn near a floor).
    // Only the short grid connection may use body clearance; grid edges use full clearance.
    const double Cell = Settings.NavigationCellSizeCm;
    // Search coordinates are relative to this path, not a fixed world boundary.
    const FVector Origin = Start;
    auto Position = [&](const FIntVector& Key) { return Origin + FVector(Key) * Cell; };
    auto KeyFor = [&](const FVector& P)
    {
        const FVector V = (P - Origin) / Cell;
        return FIntVector(FMath::RoundToInt(V.X), FMath::RoundToInt(V.Y), FMath::RoundToInt(V.Z));
    };
    struct FNode { FIntVector Key; double Cost; int32 Parent; bool Closed = false; };
    struct FOpen
    {
        double Score; int32 Index;
        bool operator<(const FOpen& Other) const
        { return Score == Other.Score ? Index > Other.Index : Score > Other.Score; }
    };
    TArray<FNode> Nodes;
    TMap<FIntVector, int32> Lookup;
    std::priority_queue<FOpen> Open;
    auto Add = [&](const FIntVector& Key, double Cost, int32 Parent)
    {
        if (int32* Existing = Lookup.Find(Key))
        {
            if (Nodes[*Existing].Closed || Nodes[*Existing].Cost <= Cost) return;
            Nodes[*Existing].Cost = Cost;
            Nodes[*Existing].Parent = Parent;
            Open.push({Cost + FVector::Dist(Position(Key), End), *Existing});
        }
        else if (Nodes.Num() < Settings.MaxNavigationNodes)
        {
            const int32 Index = Nodes.Add({Key, Cost, Parent});
            Lookup.Add(Key, Index);
            Open.push({Cost + FVector::Dist(Position(Key), End), Index});
        }
    };
    const FIntVector StartKey = KeyFor(Start);
    for (int32 X = -1; X <= 1; ++X)
        for (int32 Y = -1; Y <= 1; ++Y)
            for (int32 Z = -1; Z <= 1; ++Z)
            {
                const FIntVector Key = StartKey + FIntVector(X, Y, Z);
                const FVector P = Position(Key);
                if (IsSegmentFree(P, P, Padding) && IsSegmentFree(Start, P, 0))
                    Add(Key, FVector::Dist(Start, P), INDEX_NONE);
            }
    int32 Found = INDEX_NONE;
    int32 Closest = INDEX_NONE;
    double ClosestDistance = TNumericLimits<double>::Max();
    bool bReachedExactGoal = false;
    while (!Open.empty())
    {
        const int32 Index = Open.top().Index;
        Open.pop();
        if (Nodes[Index].Closed) continue;
        Nodes[Index].Closed = true;
        // Copy before Add can reallocate Nodes.
        const FNode Current = Nodes[Index];
        const FVector P = Position(Current.Key);
        const double Distance = FVector::Dist(P, End);
        if (Distance < ClosestDistance)
        { Closest = Index; ClosestDistance = Distance; }
        if (bEndFree && (IsSegmentFree(P, End, Padding) || (Distance <= Cell * 1.8 && IsSegmentFree(P, End, 0))))
        { Found = Index; bReachedExactGoal = true; break; }
        // A collision-adjacent objective needs an approach point, not exhaustive exploration
        // of the whole map. Deeply enclosed targets fall back to the closest explored node.
        if (bAllowPartial && !bEndFree && Distance <= Settings.ArrivalRadiusCm + Cell)
        { Found = Index; break; }
        for (int32 X = -1; X <= 1; ++X)
            for (int32 Y = -1; Y <= 1; ++Y)
                for (int32 Z = -1; Z <= 1; ++Z)
                {
                    if (!X && !Y && !Z) continue;
                    const FIntVector Key = Current.Key + FIntVector(X, Y, Z);
                    if (const int32* Existing = Lookup.Find(Key))
                        if (Nodes[*Existing].Closed) continue;
                    const FVector Next = Position(Key);
                    if (IsSegmentFree(P, Next, Padding))
                        Add(Key, Current.Cost + FVector::Dist(P, Next), Index);
                }
    }
    if (Found == INDEX_NONE && bAllowPartial) Found = Closest;
    if (Found == INDEX_NONE) return false;
    TArray<FVector> Reverse;
    for (int32 Index = Found; Index != INDEX_NONE; Index = Nodes[Index].Parent)
        Reverse.Add(Position(Nodes[Index].Key));
    for (int32 Index = Reverse.Num() - 1; Index >= 0; --Index) Path.Add(Reverse[Index]);
    if (bReachedExactGoal) Path.Add(End);
    // Greedy visibility smoothing removes unnecessary grid corners.
    FVector Previous = Start;
    for (int32 Index = 0; Index < Path.Num(); ++Index)
    {
        int32 LastVisible = Index;
        for (int32 Next = Path.Num() - 1; Next > Index; --Next)
            if (IsSegmentFree(Previous, Path[Next], Padding)) { LastVisible = Next; break; }
        if (LastVisible > Index) Path.RemoveAt(Index, LastVisible - Index);
        Previous = Path[Index];
    }
    return true;
}

FVector FIstanaSwarmSimulation::Centroid(int32 GroupId) const
{
    FVector Sum = FVector::ZeroVector;
    int32 Count = 0;
    for (const FIstanaDroneState& State : States)
    {
        if (State.GroupId == GroupId && State.bActive) { Sum += State.PositionCm; ++Count; }
    }
    return Count ? Sum / Count : FVector::ZeroVector;
}

FVector FIstanaSwarmSimulation::Target(const FGroup& Group) const
{
    return Group.Mode == EIstanaSwarmCommandType::FollowWaypoints && Group.Waypoints.IsValidIndex(Group.WaypointIndex)
        ? Group.Waypoints[Group.WaypointIndex] : Group.HoldTarget;
}

bool FIstanaSwarmSimulation::Initialize(const TArray<FIstanaSwarmConfig>& Configs, const FIstanaSwarmSettings& InSettings,
    int32 Seed, double InFixedStepSeconds, const FGuid& InRunId, FString& Error, FCollisionQuery InCollisionQuery)
{
    Error.Reset();
    if (!InRunId.IsValid() || !Positive(InFixedStepSeconds) || InFixedStepSeconds > 0.1)
    { Error = TEXT("A valid RunId and timestep in (0, 0.1] seconds are required."); return false; }
    const FIstanaSwarmSettings& S = InSettings;
    if (!Positive(S.DroneRadiusCm)
        || !Positive(S.MaxSpeedCmPerSecond) || !Positive(S.MaxAccelerationCmPerSecondSquared)
        || !Positive(S.CruiseSpeedCmPerSecond) || S.CruiseSpeedCmPerSecond > S.MaxSpeedCmPerSecond
        || !Positive(S.MaxTurnDegreesPerSecond) || S.MaxTurnDegreesPerSecond > 360
        || !Positive(S.ResponseSeconds) || !Positive(S.ArrivalRadiusCm)
        || !Positive(S.NeighborRadiusCm) || !Positive(S.SpacingCm)
        || S.SpacingCm < 2 * S.DroneRadiusCm || S.SpacingCm > S.NeighborRadiusCm
        || !Nonnegative(S.SeparationWeight) || !Nonnegative(S.AlignmentWeight) || !Nonnegative(S.CohesionWeight)
        || !Positive(S.NavigationCellSizeCm) || S.NavigationCellSizeCm < 25 || !Nonnegative(S.NavigationClearanceCm)
        || S.MaxNavigationNodes < 100 || S.MaxNavigationNodes > 100000
        || S.MaxDrones < 1 || S.MaxDrones > 256 || S.SpawnAttemptsPerDrone < 1 || S.SpawnAttemptsPerDrone > 10000)
    { Error = TEXT("Invalid swarm settings: check finite positive motion limits, navigation settings, spacing, weights and population limits."); return false; }
    FIstanaSwarmSimulation Candidate;
    Candidate.Settings = S;
    Candidate.CollisionQuery = MoveTemp(InCollisionQuery);
    Candidate.FixedStepSeconds = InFixedStepSeconds;
    Candidate.RunId = InRunId;
    TArray<FIstanaSwarmConfig> Sorted = Configs;
    Sorted.Sort([](const FIstanaSwarmConfig& A, const FIstanaSwarmConfig& B) { return A.GroupId < B.GroupId; });
    int32 PreviousGroup = INDEX_NONE;
    FRandomStream Random(Seed);
    for (const FIstanaSwarmConfig& Config : Sorted)
    {
        if (Config.GroupId < 0 || Config.GroupId == PreviousGroup || Config.DroneCount < 1
            || Config.DroneCount > S.MaxDrones - Candidate.States.Num()
            || !Finite(Config.SpawnOriginCm) || !Nonnegative(Config.SpawnRadiusCm)
            || (Config.MovementPresetId != TEXT("Stationary") && Config.MovementPresetId != TEXT("Boids")))
        { Error = TEXT("Invalid group: IDs must be unique; check counts, spawn region, and MovementPresetId (Stationary or Boids)."); return false; }
        PreviousGroup = Config.GroupId;
        FGroup Group;
        Group.Id = Config.GroupId;
        Group.CruiseSpeed = S.CruiseSpeedCmPerSecond;
        Group.Spacing = S.SpacingCm;
        Group.Mode = Config.MovementPresetId == TEXT("Stationary") ? EIstanaSwarmCommandType::Stop : EIstanaSwarmCommandType::Hold;
        const int32 FirstIndex = Candidate.States.Num();
        for (int32 Member = 0; Member < Config.DroneCount; ++Member)
        {
            bool bPlaced = false;
            for (int32 Attempt = 0; Attempt < S.SpawnAttemptsPerDrone; ++Attempt)
            {
                const FVector Position = Config.SpawnOriginCm + Random.VRand() * (Config.SpawnRadiusCm * FMath::Pow(Random.FRand(), 1.0 / 3.0));
                if (!Candidate.IsFreePosition(Position)) continue;
                bool bTooClose = false;
                for (const FIstanaDroneState& Existing : Candidate.States)
                {
                    if (FVector::DistSquared(Position, Existing.PositionCm) < FMath::Square(S.SpacingCm)) { bTooClose = true; break; }
                }
                if (bTooClose) continue;
                FIstanaDroneState State;
                State.DroneId = Candidate.States.Num();
                State.GroupId = Group.Id;
                State.PositionCm = Position;
                Candidate.States.Add(State);
                bPlaced = true;
                break;
            }
            if (!bPlaced)
            { Error = FString::Printf(TEXT("Group %d: unable to place drone %d without overlap. Enlarge the spawn region or reduce count/spacing."), Group.Id, Member); return false; }
        }
        Group.HoldTarget = Candidate.Centroid(Group.Id);
        for (int32 Index = FirstIndex; Index < Candidate.States.Num(); ++Index)
            Candidate.FormationOffsets.Add(Candidate.States[Index].PositionCm - Group.HoldTarget);
        Candidate.Groups.Add(MoveTemp(Group));
    }
    Candidate.bInitialized = true;
    Candidate.NavigationRoutes.SetNum(Candidate.States.Num());
    *this = MoveTemp(Candidate);
    return true;
}

bool FIstanaSwarmSimulation::SubmitCommand(const FIstanaSwarmCommand& Command, FString& Error)
{
    Error.Reset();
    if (!bInitialized || Command.RunId != RunId || Command.DecisionStep != Diagnostics.ExecutedSteps)
    { Error = TEXT("Command must address the initialized run and current simulation step."); return false; }
    FGroup* Group = Groups.FindByPredicate([&](const FGroup& G) { return G.Id == Command.GroupId; });
    if (!Group || Command.SequenceNumber < 0 || Command.SequenceNumber <= Group->LastSequence)
    { Error = TEXT("Unknown group or duplicate/stale command sequence."); return false; }
    switch (Command.Type)
    {
    case EIstanaSwarmCommandType::FollowWaypoints:
    {
        if (Command.WaypointsCm.IsEmpty() || Command.WaypointsCm.Num() > 256)
        { Error = TEXT("A route requires 1 to 256 waypoints."); return false; }
        for (const FVector& Point : Command.WaypointsCm)
        {
            if (!Finite(Point))
            { Error = TEXT("Waypoints must have finite coordinates."); return false; }
            if (!Command.bAllowPartialPath && !IsFreePosition(Point))
            { Error = TEXT("Strict waypoints must be clear of level collision."); return false; }
            for (int32 Index = 0; Index < States.Num(); ++Index)
            {
                if (States[Index].GroupId == Group->Id && (!Finite(Point + FormationOffsets[Index])
                    || (!Command.bAllowPartialPath && !IsFreePosition(Point + FormationOffsets[Index]))))
                { Error = TEXT("A waypoint cannot accommodate this group's formation offsets."); return false; }
            }
        }
        // Strict scripted routes validate all legs. Objective routes accept the intent;
        // unavailable paths are retried independently by each drone during stepping.
        TArray<FNavigationRoute> Proposed = NavigationRoutes;
        for (int32 Index = 0; Index < States.Num(); ++Index)
        {
            if (States[Index].GroupId != Group->Id || !States[Index].bActive) continue;
            FVector Start = States[Index].PositionCm;
            for (int32 Waypoint = 0; Waypoint < Command.WaypointsCm.Num(); ++Waypoint)
            {
                const FVector End = Command.WaypointsCm[Waypoint] + FormationOffsets[Index];
                TArray<FVector> Path;
                const bool bFoundPath = FindPath(Start, End, Path, Command.bAllowPartialPath);
                if (!bFoundPath && !Command.bAllowPartialPath)
                {
                    Error = FString::Printf(TEXT("No route found for drone %d to waypoint %d. Check collision, clearance, cell size and search budget."), States[Index].DroneId, Waypoint);
                    return false;
                }
                if (Waypoint == 0)
                {
                    Proposed[Index] = FNavigationRoute();
                    Proposed[Index].Goal = End;
                    Proposed[Index].bBlocked = !bFoundPath;
                    Proposed[Index].bPartialPath = bFoundPath && !Path.Last().Equals(End, 0.01);
                    Proposed[Index].RetryStep = Diagnostics.ExecutedSteps + 20;
                    Proposed[Index].Points = MoveTemp(Path);
                }
                if (Command.bAllowPartialPath) break;
                Start = End;
            }
            if (Command.bLoop && !Command.bAllowPartialPath)
            {
                TArray<FVector> Path;
                if (!FindPath(Start, Command.WaypointsCm[0] + FormationOffsets[Index], Path))
                { Error = TEXT("No route found for the closing leg of the loop."); return false; }
            }
        }
        NavigationRoutes = MoveTemp(Proposed);
        Group->Waypoints = Command.WaypointsCm;
        Group->WaypointIndex = 0;
        Group->bLoop = Command.bLoop;
        Group->bAllowPartialPath = Command.bAllowPartialPath;
        Group->bRouteCompleted = false;
        Group->Mode = Command.Type;
        break;
    }
    case EIstanaSwarmCommandType::Hold:
    case EIstanaSwarmCommandType::Stop:
        Group->HoldTarget = Centroid(Group->Id);
        // Holding preserves current relative positions rather than returning to spawn formation.
        for (int32 Index = 0; Index < States.Num(); ++Index)
            if (States[Index].GroupId == Group->Id)
            {
                FormationOffsets[Index] = States[Index].PositionCm - Group->HoldTarget;
                NavigationRoutes[Index] = FNavigationRoute();
            }
        Group->Waypoints.Reset();
        Group->bAllowPartialPath = false;
        Group->bRouteCompleted = false;
        Group->Mode = Command.Type;
        break;
    case EIstanaSwarmCommandType::SetCruiseSpeed:
        if (!Positive(Command.CruiseSpeedCmPerSecond) || Command.CruiseSpeedCmPerSecond > Settings.MaxSpeedCmPerSecond)
        { Error = TEXT("Cruise speed must be positive and no greater than MaxSpeed."); return false; }
        Group->CruiseSpeed = Command.CruiseSpeedCmPerSecond;
        break;
    case EIstanaSwarmCommandType::SetSpacing:
        if (!Positive(Command.SpacingCm) || Command.SpacingCm < 2 * Settings.DroneRadiusCm || Command.SpacingCm > Settings.NeighborRadiusCm)
        { Error = TEXT("Spacing must be between drone diameter and neighbor radius."); return false; }
        Group->Spacing = Command.SpacingCm;
        break;
    default: Error = TEXT("Unsupported command type."); return false;
    }
    Group->LastSequence = Command.SequenceNumber;
    return true;
}

void FIstanaSwarmSimulation::Step()
{
    if (!bInitialized) return;
    const double Dt = FixedStepSeconds;
    // Group decisions and all neighbor reads use the previous snapshot.
    for (FGroup& Group : Groups)
    {
        if (Group.Mode != EIstanaSwarmCommandType::FollowWaypoints) continue;
        bool bAllArrived = true;
        for (int32 Index = 0; Index < States.Num(); ++Index)
            if (States[Index].bActive && States[Index].GroupId == Group.Id)
                bAllArrived &= !NavigationRoutes[Index].bBlocked && !NavigationRoutes[Index].bPartialPath
                    && FVector::Dist(States[Index].PositionCm, Target(Group) + FormationOffsets[Index]) <= Settings.ArrivalRadiusCm;
        if (bAllArrived)
        {
            if (Group.WaypointIndex + 1 < Group.Waypoints.Num()) ++Group.WaypointIndex;
            else if (Group.bLoop) Group.WaypointIndex = 0;
            else
            {
                Group.HoldTarget = Group.Waypoints.Last();
                Group.Mode = EIstanaSwarmCommandType::Hold;
                Group.bRouteCompleted = true;
            }
        }
    }
    TArray<FIstanaDroneState> Next = States;
    for (int32 Index = 0; Index < States.Num(); ++Index)
    {
        const FIstanaDroneState& State = States[Index];
        if (!State.bActive) continue;
        const FGroup& Group = *Groups.FindByPredicate([&](const FGroup& G) { return G.Id == State.GroupId; });
        FVector Separation = FVector::ZeroVector;
        FVector NeighborVelocity = FVector::ZeroVector;
        FVector NeighborCenter = FVector::ZeroVector;
        int32 NeighborCount = 0;
        for (int32 OtherIndex = 0; OtherIndex < States.Num(); ++OtherIndex)
        {
            if (OtherIndex == Index || !States[OtherIndex].bActive) continue;
            const FIstanaDroneState& Other = States[OtherIndex];
            const FVector Away = State.PositionCm - Other.PositionCm;
            const double Distance = Away.Size();
            if (Distance < Group.Spacing)
                Separation += (Distance > UE_SMALL_NUMBER ? Away / Distance : FVector(Index < OtherIndex ? -1 : 1, 0, 0)) * (1.0 - Distance / Group.Spacing);
            if (Other.GroupId == Group.Id && Distance < Settings.NeighborRadiusCm)
            {
                NeighborVelocity += Other.VelocityCmPerSecond;
                NeighborCenter += Other.PositionCm;
                ++NeighborCount;
            }
        }
        const FVector Goal = Target(Group) + FormationOffsets[Index];
        FNavigationRoute& Navigation = NavigationRoutes[Index];
        FVector SteeringTarget = Goal;
        const bool bNavigate = Group.Mode != EIstanaSwarmCommandType::Stop;
        if (bNavigate)
        {
            const bool bNewGoal = !Navigation.Goal.Equals(Goal, 0.01);
            const bool bObstructed = Navigation.Points.IsValidIndex(Navigation.NextPoint)
                && !IsSegmentFree(State.PositionCm, Navigation.Points[Navigation.NextPoint], 0);
            const bool bAtPartialEnd = Navigation.bPartialPath && !Navigation.Points.IsEmpty()
                && FVector::Dist(State.PositionCm, Navigation.Points.Last()) <= Settings.ArrivalRadiusCm;
            if (bNewGoal || (Navigation.Points.IsEmpty() && !Navigation.bBlocked)
                || ((bObstructed || Navigation.bBlocked || bAtPartialEnd) && Diagnostics.ExecutedSteps >= Navigation.RetryStep))
            {
                Navigation.Goal = Goal;
                Navigation.NextPoint = 0;
                Navigation.bBlocked = !FindPath(State.PositionCm, Goal, Navigation.Points, Group.bAllowPartialPath);
                Navigation.bPartialPath = !Navigation.bBlocked && !Navigation.Points.Last().Equals(Goal, 0.01);
                Navigation.RetryStep = Diagnostics.ExecutedSteps + (Navigation.bPartialPath ? 100 : 20);
            }
            const double CornerRadius = FMath::Min(30.0, Settings.NavigationCellSizeCm * 0.15);
            while (Navigation.NextPoint + 1 < Navigation.Points.Num()
                && FVector::Dist(State.PositionCm, Navigation.Points[Navigation.NextPoint]) <= CornerRadius
                && IsSegmentFree(State.PositionCm, Navigation.Points[Navigation.NextPoint + 1], 0))
                ++Navigation.NextPoint;
            if (Navigation.Points.IsValidIndex(Navigation.NextPoint)) SteeringTarget = Navigation.Points[Navigation.NextPoint];
        }
        const FVector ToTarget = SteeringTarget - State.PositionCm;
        // Proportional arrival plus a braking-distance cap.
        const double ArrivalSpeed = FMath::Min3(Group.CruiseSpeed, ToTarget.Size() / Settings.ResponseSeconds,
            FMath::Sqrt(2.0 * Settings.MaxAccelerationCmPerSecondSquared * ToTarget.Size()));
        FVector Desired = ToTarget.GetSafeNormal() * ArrivalSpeed;
        if (NeighborCount)
        {
            Desired += (NeighborVelocity / NeighborCount - State.VelocityCmPerSecond) * Settings.AlignmentWeight;
            Desired += ((NeighborCenter / NeighborCount - State.PositionCm) / Settings.NeighborRadiusCm)
                * Group.CruiseSpeed * Settings.CohesionWeight;
        }
        Desired += Separation.GetClampedToMaxSize(1.0) * Group.CruiseSpeed * Settings.SeparationWeight;
        // The planned route supplies static-obstacle avoidance. Boid forces remain soft;
        // the final swept check guards against steering outside the planned corridor.
        if (Group.Mode == EIstanaSwarmCommandType::Stop || Navigation.bBlocked) Desired = FVector::ZeroVector;
        Desired = Desired.GetClampedToMaxSize(Group.CruiseSpeed);
        const FVector Acceleration = ((Desired - State.VelocityCmPerSecond) / Settings.ResponseSeconds)
            .GetClampedToMaxSize(Settings.MaxAccelerationCmPerSecondSquared);
        FVector Velocity = (State.VelocityCmPerSecond + Acceleration * Dt).GetClampedToMaxSize(Settings.MaxSpeedCmPerSecond);
        Velocity = LimitTurn(State.VelocityCmPerSecond, Velocity, FMath::DegreesToRadians(Settings.MaxTurnDegreesPerSecond) * Dt);
        FVector Position = State.PositionCm + Velocity * Dt;
        if (SegmentHitsObstacle(State.PositionCm, Position))
        {
            Position = State.PositionCm;
            Velocity = FVector::ZeroVector;
            ++Diagnostics.ObstacleEmergencyStops;
            Navigation.Points.Reset();
            Navigation.bBlocked = true;
        }
        Next[Index].PositionCm = Position;
        Next[Index].VelocityCmPerSecond = Velocity;
        const double Speed = Velocity.Size();
        const double ActualAcceleration = (Velocity - State.VelocityCmPerSecond).Size() / Dt;
        Diagnostics.PeakSpeedCmPerSecond = FMath::Max(Diagnostics.PeakSpeedCmPerSecond, Speed);
        Diagnostics.PeakAccelerationCmPerSecondSquared = FMath::Max(Diagnostics.PeakAccelerationCmPerSecondSquared, ActualAcceleration);
        Diagnostics.SpeedViolationSteps += Speed > Settings.MaxSpeedCmPerSecond + 0.001 ? 1 : 0;
        Diagnostics.AccelerationViolationSteps += ActualAcceleration > Settings.MaxAccelerationCmPerSecondSquared + 0.001 ? 1 : 0;
        Diagnostics.TotalPathLengthCm += FVector::Dist(Position, State.PositionCm);
    }
    States = MoveTemp(Next);
    for (int32 A = 0; A < States.Num(); ++A)
    {
        if (!States[A].bActive) continue;
        const FGroup& GroupA = *Groups.FindByPredicate([&](const FGroup& G) { return G.Id == States[A].GroupId; });
        for (int32 B = A + 1; B < States.Num(); ++B)
        {
            if (!States[B].bActive) continue;
            const FGroup& GroupB = *Groups.FindByPredicate([&](const FGroup& G) { return G.Id == States[B].GroupId; });
            const double Distance = FVector::Dist(States[A].PositionCm, States[B].PositionCm);
            Diagnostics.SpacingViolationPairSteps += Distance < FMath::Max(GroupA.Spacing, GroupB.Spacing) ? 1 : 0;
            Diagnostics.OverlapPairSteps += Distance < 2 * Settings.DroneRadiusCm ? 1 : 0;
        }
    }
    ++Diagnostics.ExecutedSteps;
    Diagnostics.SimulatedSeconds = Diagnostics.ExecutedSteps * Dt;
}

TArray<FIstanaSwarmGroupStatus> FIstanaSwarmSimulation::GetGroupStatuses() const
{
    TArray<FIstanaSwarmGroupStatus> Results;
    for (const FGroup& Group : Groups)
    {
        FIstanaSwarmGroupStatus Status;
        Status.GroupId = Group.Id;
        Status.CentroidCm = Centroid(Group.Id);
        Status.TargetCm = Target(Group);
        Status.WaypointIndex = Group.Waypoints.IsEmpty() ? INDEX_NONE : Group.WaypointIndex;
        Status.bRouteCompleted = Group.bRouteCompleted;
        Status.Mode = Group.Mode;
        for (int32 Index = 0; Index < States.Num(); ++Index)
            if (States[Index].GroupId == Group.Id && States[Index].bActive)
            {
                Status.bNavigationBlocked |= NavigationRoutes[Index].bBlocked;
                Status.bHasPartialPath |= NavigationRoutes[Index].bPartialPath;
            }
        Results.Add(Status);
    }
    return Results;
}

int64 FIstanaSwarmSimulation::GetNextCommandSequence(int32 GroupId) const
{
    const FGroup* Group = Groups.FindByPredicate([GroupId](const FGroup& G) { return G.Id == GroupId; });
    return Group && Group->LastSequence < MAX_int64 ? Group->LastSequence + 1 : INDEX_NONE;
}
