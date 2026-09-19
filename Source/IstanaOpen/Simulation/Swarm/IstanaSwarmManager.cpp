#include "Simulation/Swarm/IstanaSwarmManager.h"
#include "Components/StaticMeshComponent.h"
#include "Components/SceneComponent.h"
#include "Materials/MaterialInterface.h"
#include "Components/InputComponent.h"
#include "Engine/Engine.h"
#include "Engine/StaticMesh.h"
#include "Engine/TargetPoint.h"
#include "Engine/World.h"
#include "CollisionQueryParams.h"
#include "GameFramework/PlayerController.h"
#include "GameFramework/SpectatorPawn.h"
#include "DrawDebugHelpers.h"
#include "Simulation/Swarm/IstanaSwarmProfiling.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"
#include "InputCoreTypes.h"
#include "UObject/ConstructorHelpers.h"

AIstanaDroneVisual::AIstanaDroneVisual()
{
    PrimaryActorTick.bCanEverTick = false;
    SetRootComponent(CreateDefaultSubobject<USceneComponent>(TEXT("VisualRoot")));
    static ConstructorHelpers::FObjectFinder<UStaticMesh> Cube(TEXT("/Engine/BasicShapes/Cube.Cube"));
    static ConstructorHelpers::FObjectFinder<UStaticMesh> Sphere(TEXT("/Engine/BasicShapes/Sphere.Sphere"));
    static ConstructorHelpers::FObjectFinder<UStaticMesh> Cylinder(TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
    static ConstructorHelpers::FObjectFinder<UMaterialInterface> Paint(TEXT("/Game/Open/Materials/M_white.M_white"));
    static ConstructorHelpers::FObjectFinder<UMaterialInterface> Metal(TEXT("/Game/Open/Materials/M_metal.M_metal"));
    static ConstructorHelpers::FObjectFinder<UMaterialInterface> Dark(TEXT("/Game/Open/Materials/M_urbanroof.M_urbanroof"));
    static ConstructorHelpers::FObjectFinder<UMaterialInterface> Glass(TEXT("/Game/Open/Materials/M_glass.M_glass"));
    // Generic cosmetic quadcopter. No collision, navigation, motor physics,
    // simulation clock, sensing, random numbers or solver inputs are changed.
    int32 PartIndex = 0;
    auto Part = [&](UStaticMesh* Mesh, UMaterialInterface* Material, const FVector& Position,
                    const FVector& Size, const FRotator& Rotation = FRotator(0,0,0))
    {
        auto* Component = CreateDefaultSubobject<UStaticMeshComponent>(*FString::Printf(TEXT("DronePart%d"), PartIndex++));
        Component->SetupAttachment(RootComponent);
        Component->SetStaticMesh(Mesh);
        Component->SetMaterial(0, Material);
        Component->SetRelativeLocationAndRotation(Position, Rotation);
        Component->SetRelativeScale3D(Size / 100.);
        Component->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        Component->SetCanEverAffectNavigation(false);
        Component->bReceivesDecals = false;
        return Component;
    };
    Body = Part(Sphere.Object, Paint.Object, FVector(0,0,1), FVector(28,20,10));
    Part(Sphere.Object, Dark.Object, FVector(0,0,-2), FVector(25,18,7));
    Part(Cube.Object, Dark.Object, FVector(-3,0,5), FVector(12,11,2));
    for (double X : {-20.,20.}) for (double Y : {-20.,20.})
    {
        const double Yaw = FMath::RadiansToDegrees(FMath::Atan2(Y, X));
        Part(Cube.Object, Dark.Object, FVector(X*.55,Y*.55,0), FVector(28,3,2.5), FRotator(0,Yaw,0));
        Part(Cylinder.Object, Metal.Object, FVector(X,Y,2), FVector(5.5,5.5,5));
        Part(Cylinder.Object, Dark.Object, FVector(X,Y,5), FVector(6,6,2));
        DisplayRotors.Add(Part(Sphere.Object, Dark.Object, FVector(X,Y,6.5), FVector(27,2.8,.7), FRotator(0,Yaw+25,0)));
        Part(Cylinder.Object, Metal.Object, FVector(X,Y,7), FVector(2,2,1.5));
        Part(Cube.Object, Metal.Object, FVector(X*.55,Y*.55,-7), FVector(1.5,1.5,12));
    }
    for (double Y : {-11.,11.})
        Part(Sphere.Object, Dark.Object, FVector(0,Y,-13), FVector(34,2.8,2.8));
    // Rounded camera gimbal and inset lens provide a recognizable front.
    Part(Sphere.Object, Paint.Object, FVector(12,0,-6), FVector(8,9,8));
    Part(Cylinder.Object, Dark.Object, FVector(16,0,-6), FVector(5.5,5.5,3), FRotator(90,0,0));
    Part(Cylinder.Object, Glass.Object, FVector(17.6,0,-6), FVector(4,4,.4), FRotator(90,0,0));
}

void AIstanaDroneVisual::AnimateDisplayRotors(double PresentationSeconds)
{
    for (int32 Index = 0; Index < DisplayRotors.Num(); ++Index)
        DisplayRotors[Index]->SetRelativeRotation(FRotator(0, PresentationSeconds*937 + Index*37, 0));
}

void AIstanaDroneVisual::ApplyState(const FIstanaDroneState& State)
{
    SetActorHiddenInGame(!State.bActive);
    if (State.VelocityCmPerSecond.IsNearlyZero()) SetActorLocation(State.PositionCm);
    else
    {
        FRotator Heading = State.VelocityCmPerSecond.Rotation();
        Heading.Pitch = FMath::Clamp(Heading.Pitch, -20.0, 20.0);
        SetActorLocationAndRotation(State.PositionCm, Heading);
    }
}

AIstanaSwarmManager::AIstanaSwarmManager()
{
    PrimaryActorTick.bCanEverTick = true;
    SetRootComponent(CreateDefaultSubobject<USceneComponent>(TEXT("Root")));
    FIstanaSwarmConfig Group;
    Group.GroupId = 0;
    Group.DroneCount = 12;
    Group.SpawnOriginCm = FVector::ZeroVector;
    Group.SpawnRadiusCm = 400;
    Group.MovementPresetId = TEXT("Boids");
    Swarms.Add(Group);
}

void AIstanaSwarmManager::BeginPlay()
{
    Super::BeginPlay();
    if (bAutoInitialize) RestartDemo();
    if (bEnableDemoKeyboard && GetWorld()->GetFirstPlayerController())
    {
        EnableInput(GetWorld()->GetFirstPlayerController());
        InputComponent->BindKey(EKeys::SpaceBar, IE_Pressed, this, &AIstanaSwarmManager::TogglePaused);
        InputComponent->BindKey(EKeys::R, IE_Pressed, this, &AIstanaSwarmManager::RestartDemo);
        InputComponent->BindKey(EKeys::V, IE_Pressed, this, &AIstanaSwarmManager::ToggleDebug);
    }
}

bool AIstanaSwarmManager::InitializeSimulation(const TArray<FIstanaSwarmConfig>& Configs,
    const FIstanaSwarmSettings& InSettings, int32 InSeed, double InFixedStepSeconds, FGuid InRunId, FString& Error)
{
    if (!PrepareSimulation(Simulation, Configs, InSettings, InSeed, InFixedStepSeconds, InRunId, Error)) return false;
    AccumulatorSeconds = 0;
    DroppedWallSeconds = 0;
    LastObjective.Reset();
    bHadObjective = false;
    bHasObjectiveAttempt = false;
    ObjectiveStatus.Reset();
    DebugLabels.Reset(); DebugColors.Reset();
    for (const auto& State : Simulation.GetStates())
    {
        DebugLabels.Add(FString::Printf(TEXT("G%d / D%d"), State.GroupId, State.DroneId));
        DebugColors.Add(FLinearColor::MakeFromHSV8(uint8(State.GroupId * 67), 200, 255).ToFColor(true));
    }
    ClearVisuals();
    if (bSpawnVisuals)
    {
        for (const FIstanaDroneState& State : Simulation.GetStates())
        {
            FActorSpawnParameters Parameters;
            Parameters.Owner = this;
            Parameters.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
            Visuals.Add(GetWorld()->SpawnActor<AIstanaDroneVisual>(State.PositionCm, FRotator::ZeroRotator, Parameters));
        }
    }
    RefreshVisuals();
    return true;
}

bool AIstanaSwarmManager::PrepareSimulation(FIstanaSwarmSimulation& Candidate,
    const TArray<FIstanaSwarmConfig>& Configs, const FIstanaSwarmSettings& InSettings,
    int32 InSeed, double InFixedStepSeconds, FGuid InRunId, FString& Error)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(Istana_Spawn);
    CSV_SCOPED_TIMING_STAT(IstanaSwarm, Spawn);
    FIstanaSwarmSimulation::FCollisionQuery Query;
    FIstanaSwarmSimulation::FCollisionBatch Batch;
    if (bUseWorldCollision)
    {
        if (!GetWorld() || !GetWorld()->GetPhysicsScene())
        { Error = TEXT("World collision requires an initialized physics scene."); return false; }
        const TWeakObjectPtr<UWorld> World = GetWorld();
        const TWeakObjectPtr<AIstanaSwarmManager> Manager = this;
        const ECollisionChannel Channel = ObstacleTraceChannel;
        const bool bComplex = bTraceComplexObstacles;
        // Exact start/radius memoization exists only during a synchronous path search.
        // No frame, movement sweep, or later search can reuse stale collision results.
        struct FOverlapMemo { bool bActive = false; TMap<TPair<FVector, double>, bool> Values; };
        const auto Memo = MakeShared<FOverlapMemo>();
        Batch = [Memo](bool bActive) { Memo->bActive = bActive; Memo->Values.Reset(); };
        Query = [World, Manager, Channel, bComplex, Memo](const FVector& Start, const FVector& End, double Radius)
        {
            TRACE_CPUPROFILER_EVENT_SCOPE(Istana_Collision);
            if (!World.IsValid()) return true;
            FCollisionQueryParams Params(SCENE_QUERY_STAT(IstanaSwarmNavigation), bComplex);
            if (Manager.IsValid()) Params.AddIgnoredActor(Manager.Get());
            const FCollisionShape Shape = FCollisionShape::MakeSphere(float(Radius));
            bool bOverlap;
            const TPair<FVector, double> Key(Start, Radius);
            const bool* Cached = Memo->bActive ? Memo->Values.Find(Key) : nullptr;
            if (Cached) bOverlap = *Cached;
            else
            {
                TRACE_CPUPROFILER_EVENT_SCOPE(Istana_Overlap);
                bOverlap = World->OverlapBlockingTestByChannel(Start, FQuat::Identity, Channel, Shape, Params);
                if (Memo->bActive) Memo->Values.Add(Key, bOverlap);
            }
            if (bOverlap) return true;
            TRACE_CPUPROFILER_EVENT_SCOPE(Istana_Sweep);
            FHitResult Hit;
            return World->SweepSingleByChannel(Hit, Start, End, FQuat::Identity, Channel, Shape, Params);
        };
    }
    return Candidate.Initialize(Configs, InSettings, InSeed, InFixedStepSeconds, InRunId, Error, MoveTemp(Query), MoveTemp(Batch));
}

bool AIstanaSwarmManager::ResetSimulation(FString& Error)
{
    TArray<FIstanaSwarmConfig> RuntimeSwarms = Swarms;
    if (bSpawnRelativeToManager)
        for (FIstanaSwarmConfig& Group : RuntimeSwarms)
            Group.SpawnOriginCm = GetActorTransform().TransformPositionNoScale(Group.SpawnOriginCm);
    FIstanaSwarmSettings RuntimeSettings = MovementPreset ? MovementPreset->Settings : Settings;
    return InitializeSimulation(RuntimeSwarms, RuntimeSettings,
        Seed, FixedStepSeconds, FGuid::NewGuid(), Error);
}

void AIstanaSwarmManager::RestartDemo()
{
    FString Error;
    if (!ResetSimulation(Error))
    {
        UE_LOG(LogTemp, Error, TEXT("Swarm reset rejected: %s"), *Error);
        ObjectiveStatus = Error;
        if (GEngine) GEngine->AddOnScreenDebugMessage(uint64(GetUniqueID()), 15, FColor::Red, Error);
        return;
    }
    if (bFollowObjective && IsValid(ObjectiveTarget))
    {
        UpdateObjective();
        return;
    }
    if (!DemoWaypointsCm.IsEmpty())
    {
        for (const FIstanaSwarmGroupStatus& Group : GetGroupStatuses())
        {
            FIstanaSwarmCommand Command;
            Command.RunId = GetRunId();
            Command.GroupId = Group.GroupId;
            Command.Type = EIstanaSwarmCommandType::FollowWaypoints;
            Command.WaypointsCm = DemoWaypointsCm;
            Command.bLoop = bLoopDemoRoute;
            if (!SubmitCommand(Command, Error)) UE_LOG(LogTemp, Error, TEXT("Demo route rejected: %s"), *Error);
        }
    }
}

bool AIstanaSwarmManager::SubmitCommand(const FIstanaSwarmCommand& Command, FString& Error)
{
    return Simulation.SubmitCommand(Command, Error);
}

void AIstanaSwarmManager::AdvanceOneStep()
{
    if (!Simulation.IsInitialized()) return;
    UpdateObjective();
    Simulation.Step();
    RefreshVisuals();
    OnSimulationStepped.Broadcast(Simulation.GetDiagnostics().ExecutedSteps);
}

bool AIstanaSwarmManager::CommandAllGroups(EIstanaSwarmCommandType Type, const FVector& Position, FString& Error)
{
    // Validate the entire update on a copy: never send half the groups to a new point.
    FIstanaSwarmSimulation Candidate = Simulation;
    for (const FIstanaSwarmGroupStatus& Group : Candidate.GetGroupStatuses())
    {
        FIstanaSwarmCommand Command;
        Command.RunId = Candidate.GetRunId();
        Command.DecisionStep = Candidate.GetDiagnostics().ExecutedSteps;
        Command.SequenceNumber = Candidate.GetNextCommandSequence(Group.GroupId);
        Command.GroupId = Group.GroupId;
        Command.Type = Type;
        if (Type == EIstanaSwarmCommandType::FollowWaypoints)
        {
            Command.WaypointsCm.Add(Position);
            Command.bAllowPartialPath = true;
        }
        if (!Candidate.SubmitCommand(Command, Error)) return false;
    }
    Simulation = MoveTemp(Candidate);
    Error.Reset();
    return true;
}

void AIstanaSwarmManager::UpdateObjective()
{
    if (!bFollowObjective)
    {
        bHasObjectiveAttempt = false;
        bHadObjective = false;
        ObjectiveStatus = TEXT("Automatic objective following disabled.");
        return;
    }
    if (!IsValid(ObjectiveTarget))
    {
        if (bHadObjective)
        {
            FString Error;
            const bool bStopped = CommandAllGroups(EIstanaSwarmCommandType::Stop, FVector::ZeroVector, Error);
            ObjectiveStatus = bStopped ? TEXT("Objective removed; groups braking to a stop.") : Error;
        }
        bHadObjective = false;
        bHasObjectiveAttempt = false;
        LastObjective.Reset();
        return;
    }
    const FVector Position = ObjectiveTarget->GetActorLocation();
    if (bHasObjectiveAttempt && LastObjective.Get() == ObjectiveTarget.Get()
        && Position.Equals(LastObjectivePosition, 1.0)) return;
    bHadObjective = true;
    bHasObjectiveAttempt = true;
    LastObjective = ObjectiveTarget;
    LastObjectivePosition = Position;
    FString Error;
    if (CommandAllGroups(EIstanaSwarmCommandType::FollowWaypoints, Position, Error))
    {
        ObjectiveStatus = TEXT("Following objective; routing around collision and approaching reachable space near blocked targets.");
    }
    else
    {
        const FString Rejection = Error;
        const bool bStopped = CommandAllGroups(EIstanaSwarmCommandType::Stop, FVector::ZeroVector, Error);
        ObjectiveStatus = FString::Printf(TEXT("Objective rejected: %s %s"), *Rejection,
            bStopped ? TEXT("Groups braking to a stop.") : *Error);
        UE_LOG(LogTemp, Warning, TEXT("%s"), *ObjectiveStatus);
    }
}

void AIstanaSwarmManager::Tick(float DeltaSeconds)
{
    CSV_SCOPED_TIMING_STAT(IstanaSwarm, ManagerTick);
    Super::Tick(DeltaSeconds);
    if (Simulation.IsInitialized() && bAutoAdvance && !bPaused && FMath::IsFinite(DeltaSeconds) && DeltaSeconds > 0)
    {
        const double Dt = Simulation.GetFixedStepSeconds();
        const double Limit = Dt * FMath::Clamp(MaxStepsPerFrame, 1, 64);
        const double Requested = AccumulatorSeconds + DeltaSeconds;
        DroppedWallSeconds += FMath::Max(0.0, Requested - Limit);
        AccumulatorSeconds = FMath::Min(Requested, Limit);
        while (AccumulatorSeconds + UE_SMALL_NUMBER >= Dt)
        {
            AccumulatorSeconds = FMath::Max(0.0, AccumulatorSeconds - Dt);
            AdvanceOneStep();
        }
    }
    if (bDrawDebug && Simulation.IsInitialized()) DrawDiagnostics();
    CSV_CUSTOM_STAT(IstanaSwarm, DroppedWallSeconds, DroppedWallSeconds, ECsvCustomStatOp::Set);
}

void AIstanaSwarmManager::RefreshVisuals()
{
    TRACE_CPUPROFILER_EVENT_SCOPE(Istana_Visuals);
    CSV_SCOPED_TIMING_STAT(IstanaSwarm, Visuals);
    const TArray<FIstanaDroneState>& States = Simulation.GetStates();
    for (int32 Index = 0; Index < Visuals.Num() && Index < States.Num(); ++Index)
        if (IsValid(Visuals[Index])) Visuals[Index]->ApplyState(States[Index]);
}

void AIstanaSwarmManager::ClearVisuals()
{
    for (AIstanaDroneVisual* Visual : Visuals) if (IsValid(Visual)) Visual->Destroy();
    Visuals.Reset();
}

void AIstanaSwarmManager::EndPlay(const EEndPlayReason::Type Reason)
{
    ClearVisuals();
    Super::EndPlay(Reason);
}

void AIstanaSwarmManager::Destroyed()
{
    ClearVisuals();
    Super::Destroyed();
}

TArray<FIstanaDroneState> AIstanaSwarmManager::GetDroneStates() const { return Simulation.GetStates(); }
TArray<FIstanaSwarmGroupStatus> AIstanaSwarmManager::GetGroupStatuses() const { return Simulation.GetGroupStatuses(); }
FIstanaSwarmDiagnostics AIstanaSwarmManager::GetDiagnostics() const { return Simulation.GetDiagnostics(); }
FIstanaSwarmWorkCounters AIstanaSwarmManager::GetWorkCounters() const { return Simulation.GetWorkCounters(); }
FGuid AIstanaSwarmManager::GetRunId() const { return Simulation.GetRunId(); }
void AIstanaSwarmManager::TogglePaused() { bPaused = !bPaused; AccumulatorSeconds = 0; }
void AIstanaSwarmManager::ToggleDebug() { bDrawDebug = !bDrawDebug; }

void AIstanaSwarmManager::DrawDiagnostics() const
{
    TRACE_CPUPROFILER_EVENT_SCOPE(Istana_Debug);
    CSV_SCOPED_TIMING_STAT(IstanaSwarm, Debug);
    const auto Statuses = Simulation.GetGroupStatuses();
    const FIstanaSwarmSettings& S = Simulation.GetSettings();
    for (const FIstanaDroneState& State : Simulation.GetStates())
    {
        if (!State.bActive) continue;
        const FColor Color = DebugColors[State.DroneId];
        DrawDebugDirectionalArrow(GetWorld(), State.PositionCm, State.PositionCm + State.VelocityCmPerSecond * 0.4, 20, Color);
        DrawDebugString(GetWorld(), State.PositionCm + FVector(0, 0, 35), DebugLabels[State.DroneId], nullptr, Color, 0);
    }
    for (const FIstanaSwarmGroupStatus& Group : Statuses)
    {
        FBox GroupBounds(ForceInit);
        bool bFirst = true;
        for (const FIstanaDroneState& State : Simulation.GetStates())
        {
            if (State.GroupId != Group.GroupId || !State.bActive) continue;
            GroupBounds += State.PositionCm;
            if (bFirst) DrawDebugSphere(GetWorld(), State.PositionCm, S.NeighborRadiusCm, 16, FColor(90, 90, 90));
            bFirst = false;
        }
        if (GroupBounds.IsValid) DrawDebugBox(GetWorld(), GroupBounds.GetCenter(), GroupBounds.GetExtent() + FVector(S.DroneRadiusCm), FColor::Green);
        DrawDebugLine(GetWorld(), Group.CentroidCm, Group.TargetCm, FColor::Yellow);
        DrawDebugSphere(GetWorld(), Group.TargetCm, S.ArrivalRadiusCm, 16, FColor::Yellow);
    }
    if (GEngine)
    {
        const FIstanaSwarmDiagnostics& D = Simulation.GetDiagnostics();
        int32 BlockedGroups = 0;
        for (const FIstanaSwarmGroupStatus& Group : Statuses)
            if (Group.bNavigationBlocked) ++BlockedGroups;
        GEngine->AddOnScreenDebugMessage(uint64(GetUniqueID()), 0, FColor::White,
            FString::Printf(TEXT("SWARM %s | %d drones | %.1fs | blocked groups %d | overlaps %lld | emergency stops %lld | SPACE pause, R reset, V debug\n%s"),
                bPaused ? TEXT("PAUSED") : TEXT("RUNNING"), Simulation.GetStates().Num(), D.SimulatedSeconds,
                BlockedGroups, D.OverlapPairSteps, D.ObstacleEmergencyStops, *ObjectiveStatus));
    }
}

AIstanaSwarmDemoGameMode::AIstanaSwarmDemoGameMode()
{
    DefaultPawnClass = ASpectatorPawn::StaticClass();
}
