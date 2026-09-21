#include "Simulation/Swarm/IstanaSwarmSimulation.h"
#include "Simulation/Swarm/IstanaSwarmManager.h"
#include "Simulation/RedTeam/RedTeamManager.h"
#include "Engine/TargetPoint.h"
#include "EngineUtils.h"
#include "Misc/AutomationTest.h"
#include "HAL/PlatformTime.h"
#include "Engine/World.h"
#include "Engine/Engine.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include <limits>

#if WITH_DEV_AUTOMATION_TESTS
namespace
{
    FIstanaSwarmConfig MakeGroup(int32 Id = 0, int32 Count = 1, FVector Origin = FVector(0, 0, 700), double Radius = 0)
    {
        FIstanaSwarmConfig G;
        G.GroupId = Id;
        G.DroneCount = Count;
        G.SpawnOriginCm = Origin;
        G.SpawnRadiusCm = Radius;
        G.MovementPresetId = TEXT("Boids");
        return G;
    }
    FIstanaSwarmCommand Route(const FIstanaSwarmSimulation& Sim, const FVector& Point, int32 GroupId = 0, int64 Sequence = 0)
    {
        FIstanaSwarmCommand C;
        C.RunId = Sim.GetRunId();
        C.DecisionStep = Sim.GetDiagnostics().ExecutedSteps;
        C.SequenceNumber = Sequence;
        C.GroupId = GroupId;
        C.Type = EIstanaSwarmCommandType::FollowWaypoints;
        C.WaypointsCm.Add(Point);
        return C;
    }
    bool SameStates(const TArray<FIstanaDroneState>& A, const TArray<FIstanaDroneState>& B)
    {
        if (A.Num() != B.Num()) return false;
        for (int32 I = 0; I < A.Num(); ++I)
            if (A[I].DroneId != B[I].DroneId || A[I].GroupId != B[I].GroupId
                || A[I].PositionCm != B[I].PositionCm || A[I].VelocityCmPerSecond != B[I].VelocityCmPerSecond) return false;
        return true;
    }
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIstanaSwarmSpawnTest, "Istana.Simulation.Swarm.SpawnAndReset",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FIstanaSwarmSpawnTest::RunTest(const FString& Parameters)
{
    FIstanaSwarmSimulation A, B;
    FIstanaSwarmSettings Settings;
    const FGuid Run(1, 2, 3, 4);
    FString Error;
    const FIstanaSwarmConfig Left = MakeGroup(2, 8, FVector(-1200, 0, 700), 400);
    const FIstanaSwarmConfig Right = MakeGroup(7, 8, FVector(1200, 0, 700), 400);
    TestTrue(TEXT("Spawn valid groups"), A.Initialize({Left, Right}, Settings, 123, 0.05, Run, Error));
    TestTrue(TEXT("Reversed config order"), B.Initialize({Right, Left}, Settings, 123, 0.05, Run, Error));
    TestTrue(TEXT("Order-independent seeded spawning"), SameStates(A.GetStates(), B.GetStates()));
    TestEqual(TEXT("All members spawned"), A.GetStates().Num(), 16);
    for (int32 I = 0; I < A.GetStates().Num(); ++I)
        for (int32 J = I + 1; J < A.GetStates().Num(); ++J)
            TestTrue(TEXT("Initial spacing respected"), FVector::Dist(A.GetStates()[I].PositionCm, A.GetStates()[J].PositionCm) >= Settings.SpacingCm);
    for (int32 Step = 0; Step < 100; ++Step) { A.Step(); B.Step(); }
    TestTrue(TEXT("Repeatable stepping"), SameStates(A.GetStates(), B.GetStates()));
    TestTrue(TEXT("Reset"), A.Initialize({Left, Right}, Settings, 123, 0.05, FGuid::NewGuid(), Error));
    TestTrue(TEXT("Fresh equivalent"), B.Initialize({Right, Left}, Settings, 123, 0.05, Run, Error));
    TestTrue(TEXT("Run identity does not change RNG"), SameStates(A.GetStates(), B.GetStates()));
    TestEqual(TEXT("Metrics reset"), A.GetDiagnostics().ExecutedSteps, int64(0));
    const TArray<FIstanaDroneState> Before = A.GetStates();
    TestFalse(TEXT("Impossible zero-radius group rejected"), A.Initialize({MakeGroup(0, 2)}, Settings, 0, 0.05, Run, Error));
    TestTrue(TEXT("Failed reset retains prior state"), SameStates(Before, A.GetStates()));
    TestFalse(TEXT("Duplicate group rejected"), A.Initialize({Left, Left}, Settings, 0, 0.05, Run, Error));
    Settings.MaxSpeedCmPerSecond = std::numeric_limits<double>::quiet_NaN();
    TestFalse(TEXT("Nonfinite tuning rejected"), A.Initialize({Left}, Settings, 0, 0.05, Run, Error));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIstanaSwarmCommandsTest, "Istana.Simulation.Swarm.CommandsAndArrival",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FIstanaSwarmCommandsTest::RunTest(const FString& Parameters)
{
    FIstanaSwarmSimulation Sim;
    FString Error;
    TestTrue(TEXT("Initialize"), Sim.Initialize({MakeGroup()}, FIstanaSwarmSettings(), 1, 0.05, FGuid::NewGuid(), Error));
    FIstanaSwarmCommand C = Route(Sim, FVector(1200, 0, 700));
    TestTrue(TEXT("Route accepted"), Sim.SubmitCommand(C, Error));
    TestFalse(TEXT("Duplicate rejected"), Sim.SubmitCommand(C, Error));
    C.SequenceNumber = 1;
    C.WaypointsCm[0].X = std::numeric_limits<double>::infinity();
    TestFalse(TEXT("Nonfinite target rejected"), Sim.SubmitCommand(C, Error));
    for (int32 Step = 0; Step < 240; ++Step) Sim.Step();
    TestTrue(TEXT("Finite route completed"), Sim.GetGroupStatuses()[0].bRouteCompleted);
    TestTrue(TEXT("Settled at target"), FVector::Dist(Sim.GetStates()[0].PositionCm, FVector(1200, 0, 700)) < 5);
    TestEqual(TEXT("Speed limit"), Sim.GetDiagnostics().SpeedViolationSteps, int64(0));
    TestEqual(TEXT("Acceleration limit"), Sim.GetDiagnostics().AccelerationViolationSteps, int64(0));
    C = Route(Sim, FVector(0, 1200, 700), 0, 2);
    TestTrue(TEXT("Second route"), Sim.SubmitCommand(C, Error));
    for (int32 Step = 0; Step < 30; ++Step) Sim.Step();
    C.Type = EIstanaSwarmCommandType::Stop;
    C.SequenceNumber = 3;
    TestFalse(TEXT("Stale decision rejected"), Sim.SubmitCommand(C, Error));
    C.DecisionStep = Sim.GetDiagnostics().ExecutedSteps;
    TestTrue(TEXT("Stop accepted"), Sim.SubmitCommand(C, Error));
    for (int32 Step = 0; Step < 120; ++Step) Sim.Step();
    TestTrue(TEXT("Stop brakes to rest"), Sim.GetStates()[0].VelocityCmPerSecond.Size() < 0.01);
    C.SequenceNumber = 4;
    C.DecisionStep = Sim.GetDiagnostics().ExecutedSteps;
    C.RunId = FGuid::NewGuid();
    TestFalse(TEXT("Different run rejected"), Sim.SubmitCommand(C, Error));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIstanaSwarmSteeringTest, "Istana.Simulation.Swarm.SteeringAndCollision",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FIstanaSwarmSteeringTest::RunTest(const FString& Parameters)
{
    FIstanaSwarmSettings Settings;
    Settings.CohesionWeight = 0;
    Settings.AlignmentWeight = 0;
    FIstanaSwarmSimulation Sim;
    FString Error;
    TestTrue(TEXT("Pair initialized"), Sim.Initialize({MakeGroup(0, 2, FVector(0, 0, 700), 150)}, Settings, 1, 0.05, FGuid::NewGuid(), Error));
    const double Before = FVector::Dist(Sim.GetStates()[0].PositionCm, Sim.GetStates()[1].PositionCm);
    FIstanaSwarmCommand C;
    C.RunId = Sim.GetRunId();
    C.GroupId = 0;
    C.Type = EIstanaSwarmCommandType::SetSpacing;
    C.SpacingCm = 500;
    TestTrue(TEXT("Spacing command"), Sim.SubmitCommand(C, Error));
    for (int32 Step = 0; Step < 30; ++Step) Sim.Step();
    TestTrue(TEXT("Separation increases spacing"), FVector::Dist(Sim.GetStates()[0].PositionCm, Sim.GetStates()[1].PositionCm) > Before);

    TestTrue(TEXT("Single drone reset"), Sim.Initialize({MakeGroup()}, Settings, 0, 0.05, FGuid::NewGuid(), Error));
    TestTrue(TEXT("Route"), Sim.SubmitCommand(Route(Sim, FVector(2000, 0, 700)), Error));
    for (int32 Step = 0; Step < 50; ++Step) Sim.Step();
    TestTrue(TEXT("Reverse route"), Sim.SubmitCommand(Route(Sim, FVector(-2000, 0, 700), 0, 1), Error));
    bool bTurnBounded = true;
    for (int32 Step = 0; Step < 100; ++Step)
    {
        const FVector Previous = Sim.GetStates()[0].VelocityCmPerSecond;
        Sim.Step();
        const FVector Next = Sim.GetStates()[0].VelocityCmPerSecond;
        if (Previous.Size() > 0.01 && Next.Size() > 0.01)
        {
            const double Angle = FMath::RadiansToDegrees(FMath::Acos(FMath::Clamp(FVector::DotProduct(Previous.GetSafeNormal(), Next.GetSafeNormal()), -1.0, 1.0)));
            bTurnBounded &= Angle <= Settings.MaxTurnDegreesPerSecond * 0.05 + 0.001;
        }
    }
    TestTrue(TEXT("Turn rate bounded"), bTurnBounded);
    TestEqual(TEXT("Normal acceleration bounded"), Sim.GetDiagnostics().AccelerationViolationSteps, int64(0));
    const FVector ObstacleCenter(1000, 0, 700);
    const double ObstacleRadius = 200;
    auto Collision = [ObstacleCenter, ObstacleRadius](const FVector& A, const FVector& B, double Radius)
    {
        return FVector::Dist(FMath::ClosestPointOnSegment(ObstacleCenter, A, B), ObstacleCenter) <= ObstacleRadius + Radius;
    };
    TestTrue(TEXT("Collision environment"), Sim.Initialize({MakeGroup()}, Settings, 0, 0.05, FGuid::NewGuid(), Error, Collision));
    TestTrue(TEXT("Route across obstacle"), Sim.SubmitCommand(Route(Sim, FVector(2000, 0, 700)), Error));
    bool bOutsideObstacle = true;
    for (int32 Step = 0; Step < 400; ++Step)
    {
        Sim.Step();
        const FVector P = Sim.GetStates()[0].PositionCm;
        bOutsideObstacle &= FVector::Dist(P, ObstacleCenter) > ObstacleRadius + Settings.DroneRadiusCm;
    }
    TestTrue(TEXT("Obstacle nonpenetration"), bOutsideObstacle);
    TestTrue(TEXT("Arrives beyond the blocking obstacle"), Sim.GetGroupStatuses()[0].bRouteCompleted);
    TestTrue(TEXT("Settles at destination beyond obstacle"), FVector::Dist(Sim.GetStates()[0].PositionCm, FVector(2000, 0, 700)) < 5);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIstanaSwarmFlightEnvelopeTest, "Istana.Simulation.Swarm.MavicFlightEnvelope",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FIstanaSwarmFlightEnvelopeTest::RunTest(const FString& Parameters)
{
    FIstanaSwarmSettings Settings;
    Settings.WindVelocityCmPerSecond = FVector(300, -150, 0);
    FIstanaSwarmSimulation Sim;
    FString Error;
    TestTrue(TEXT("Flight-envelope simulation initialized"),
        Sim.Initialize({MakeGroup()}, Settings, 22, 0.05, FGuid::NewGuid(), Error));
    TestTrue(TEXT("Three-axis route accepted"),
        Sim.SubmitCommand(Route(Sim, FVector(10000, 5000, 8000)), Error));

    FVector PreviousVelocity = Sim.GetStates()[0].VelocityCmPerSecond;
    FVector PreviousAcceleration = FVector::ZeroVector;
    bool bVelocityBounded = true;
    bool bAccelerationBounded = true;
    bool bJerkBounded = true;
    for (int32 Step = 0; Step < 300; ++Step)
    {
        Sim.Step();
        const FVector Velocity = Sim.GetStates()[0].VelocityCmPerSecond;
        const FVector AirVelocity = Velocity - Settings.WindVelocityCmPerSecond;
        const FVector Acceleration = (Velocity - PreviousVelocity) / 0.05;
        const FVector Jerk = (Acceleration - PreviousAcceleration) / 0.05;
        bVelocityBounded &= FVector2D(AirVelocity.X, AirVelocity.Y).Size() <= Settings.MaxSpeedCmPerSecond + 0.001;
        bVelocityBounded &= AirVelocity.Z <= Settings.MaxAscentSpeedCmPerSecond + 0.001;
        bVelocityBounded &= AirVelocity.Z >= -Settings.MaxDescentSpeedCmPerSecond - 0.001;
        bAccelerationBounded &= Acceleration.Size() <= Settings.MaxAccelerationCmPerSecondSquared + 0.01;
        // The first sample includes the initialized wind-relative state; subsequent samples test the controller.
        if (Step > 0) bJerkBounded &= Jerk.Size() <= Settings.MaxJerkCmPerSecondCubed + 0.1;
        PreviousVelocity = Velocity;
        PreviousAcceleration = Acceleration;
    }
    TestTrue(TEXT("Air-relative horizontal and vertical speeds bounded"), bVelocityBounded);
    TestTrue(TEXT("Tilt-derived acceleration bounded"), bAccelerationBounded);
    TestTrue(TEXT("Commanded motion is jerk bounded"), bJerkBounded);
    TestEqual(TEXT("No internal speed-envelope violations"), Sim.GetDiagnostics().SpeedViolationSteps, int64(0));
    TestEqual(TEXT("No internal acceleration-envelope violations"), Sim.GetDiagnostics().AccelerationViolationSteps, int64(0));
    TestEqual(TEXT("No internal jerk-envelope violations"), Sim.GetDiagnostics().JerkViolationSteps, int64(0));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIstanaSwarmNavigationTest, "Istana.Simulation.Swarm.Navigation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FIstanaSwarmNavigationTest::RunTest(const FString& Parameters)
{
    FIstanaSwarmSimulation Sim, Repeat;
    FIstanaSwarmSettings Settings;
    FString Error;
    const FGuid Run(1, 2, 3, 4);
    // A box wall supplied through the same collision-query interface as the world adapter.
    const FBox Wall(FVector(700, -600, 100), FVector(1100, 600, 1500));
    auto Blocked = [Wall](const FVector& A, const FVector& B, double Radius)
    {
        const FBox Expanded = Wall.ExpandBy(Radius);
        return Expanded.IsInsideOrOn(A) || Expanded.IsInsideOrOn(B)
            || FMath::LineBoxIntersection(Expanded, A, B, B - A);
    };
    const FIstanaSwarmConfig Group = MakeGroup(0, 12, FVector(-500, 0, 700), 350);
    TestTrue(TEXT("Group initialized with collision query"), Sim.Initialize({Group}, Settings, 42, 0.05, Run, Error, Blocked));
    TestTrue(TEXT("Repeat initialized"), Repeat.Initialize({Group}, Settings, 42, 0.05, Run, Error, Blocked));
    const FVector Goal(2200, 0, 700);
    TestTrue(TEXT("Wall detour accepted"), Sim.SubmitCommand(Route(Sim, Goal), Error));
    TestTrue(TEXT("Repeat route"), Repeat.SubmitCommand(Route(Repeat, Goal), Error));
    bool bClear = true;
    for (int32 Step = 0; Step < 1600; ++Step)
    {
        const TArray<FIstanaDroneState> Before = Sim.GetStates();
        Sim.Step(); Repeat.Step();
        for (int32 Index = 0; Index < Before.Num(); ++Index)
            bClear &= !Blocked(Before[Index].PositionCm, Sim.GetStates()[Index].PositionCm, Settings.DroneRadiusCm);
    }
    TestTrue(TEXT("Whole swarm reaches destination around wall"), Sim.GetGroupStatuses()[0].bRouteCompleted);
    TestTrue(TEXT("No swept wall penetration"), bClear);
    TestTrue(TEXT("Navigation repeats deterministically"), SameStates(Sim.GetStates(), Repeat.GetStates()));

    // An infinite slab blocks the route; termination comes from the search budget, not an arena.
    auto Sealed = [](const FVector& A, const FVector& B, double Radius)
    {
        return FMath::Max(A.X, B.X) >= 700 - Radius && FMath::Min(A.X, B.X) <= 1100 + Radius;
    };
    TestTrue(TEXT("Infinite wall environment initialized"), Sim.Initialize({MakeGroup()}, Settings, 1, 0.05, Run, Error, Sealed));
    TestFalse(TEXT("Unreachable destination rejected"), Sim.SubmitCommand(Route(Sim, Goal), Error));
    TestEqual(TEXT("Rejected route preserves sequence"), Sim.GetNextCommandSequence(0), int64(0));
    TestEqual(TEXT("Rejected route preserves mode"), Sim.GetGroupStatuses()[0].Mode, EIstanaSwarmCommandType::Hold);
    TestFalse(TEXT("Spawn inside wall rejected"), Sim.Initialize({MakeGroup(0, 1, FVector(900, 0, 700))}, Settings, 1, 0.05, Run, Error, Sealed));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIstanaSwarmPopulationTest, "Istana.Simulation.Swarm.Population",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FIstanaSwarmPopulationTest::RunTest(const FString& Parameters)
{
    FIstanaSwarmSimulation Sim;
    FString Error;
    FIstanaSwarmSettings Settings;
    TestTrue(TEXT("128 drones initialized"), Sim.Initialize({MakeGroup(0, 128, FVector::ZeroVector, 2000)}, Settings, 12, 0.05, FGuid::NewGuid(), Error));
    const double Start = FPlatformTime::Seconds();
    for (int32 Step = 0; Step < 100; ++Step) Sim.Step();
    AddInfo(FString::Printf(TEXT("128 drones, 100 value-only steps: %.3f seconds (not a hardware-independent performance guarantee)."), FPlatformTime::Seconds() - Start));
    TestEqual(TEXT("All states retained"), Sim.GetStates().Num(), 128);
    TestEqual(TEXT("No speed violations"), Sim.GetDiagnostics().SpeedViolationSteps, int64(0));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIstanaSwarmWorldTest, "Istana.Simulation.Swarm.WorldCollisionAndPlacement",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FIstanaSwarmWorldTest::RunTest(const FString& Parameters)
{
    const UWorld::InitializationValues Values = UWorld::InitializationValues()
        .AllowAudioPlayback(false).CreatePhysicsScene(true).CreateNavigation(false)
        .CreateAISystem(false).ShouldSimulatePhysics(false);
    UWorld* World = UWorld::CreateWorld(EWorldType::Game, false, NAME_None, nullptr, true, ERHIFeatureLevel::Num, &Values);
    if (!TestNotNull(TEXT("Test world"), World)) return false;
    GEngine->CreateNewWorldContext(EWorldType::Game).SetCurrentWorld(World);
    AIstanaSwarmManager* Manager = World->SpawnActor<AIstanaSwarmManager>();
    Manager->bAutoAdvance = false;
    Manager->bAutoInitialize = false;
    Manager->bFollowObjective = false;
    Manager->bSpawnVisuals = false;
    Manager->SetActorLocation(FVector(10000, 20000, 3000));
    Manager->Swarms = {MakeGroup(0, 1, FVector::ZeroVector)};
    FString Error;
    if (!TestTrue(TEXT("Placed manager initialized"), Manager->ResetSimulation(Error)))
    { AddError(Error); GEngine->DestroyWorldContext(World); World->DestroyWorld(false); return false; }
    TestEqual(TEXT("Spawn at manager"), Manager->GetDroneStates()[0].PositionCm, Manager->GetActorLocation());
    Manager->SetActorLocation(FVector(11000, 21000, 3500));
    TestTrue(TEXT("Moved manager reset"), Manager->ResetSimulation(Error));
    TestEqual(TEXT("Reset follows moved manager"), Manager->GetDroneStates()[0].PositionCm, Manager->GetActorLocation());

    AActor* Wall = World->SpawnActor<AActor>();
    UStaticMeshComponent* Box = NewObject<UStaticMeshComponent>(Wall);
    Wall->SetRootComponent(Box);
    Box->SetStaticMesh(LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube")));
    Box->SetWorldScale3D(FVector(4, 12, 18));
    Box->SetCollisionProfileName(TEXT("BlockAll"));
    Box->RegisterComponent();
    Wall->SetActorLocation(FVector(12000, 21000, 3500));
    World->Tick(LEVELTICK_All, 0.05f);
    FHitResult Probe;
    TestTrue(TEXT("World collision probe sees wall"), World->SweepSingleByChannel(Probe,
        FVector(11000, 21000, 3500), FVector(13200, 21000, 3500), FQuat::Identity,
        ECC_Visibility, FCollisionShape::MakeSphere(20)));
    FIstanaSwarmCommand Command;
    Command.RunId = Manager->GetRunId();
    Command.GroupId = 0;
    Command.Type = EIstanaSwarmCommandType::FollowWaypoints;
    Command.WaypointsCm = {FVector(13200, 21000, 3500)};
    TestTrue(TEXT("World wall route accepted"), Manager->SubmitCommand(Command, Error));
    bool bClear = true;
    for (int32 Step = 0; Step < 1000; ++Step)
    {
        const FVector Before = Manager->GetDroneStates()[0].PositionCm;
        Manager->AdvanceOneStep();
        const FVector After = Manager->GetDroneStates()[0].PositionCm;
        FHitResult Hit;
        bClear &= !World->SweepSingleByChannel(Hit, Before, After, FQuat::Identity, ECC_Visibility, FCollisionShape::MakeSphere(20));
    }
    TestTrue(TEXT("World wall never penetrated"), bClear);
    TestTrue(TEXT("Arrived around world wall"), Manager->GetGroupStatuses()[0].bRouteCompleted);
    Command.SequenceNumber = 1;
    Command.DecisionStep = Manager->GetDiagnostics().ExecutedSteps;
    Command.WaypointsCm = {Wall->GetActorLocation()};
    TestFalse(TEXT("Destination inside real collision rejected"), Manager->SubmitCommand(Command, Error));

    // The same destination is accepted by objective following, without a collision bypass.
    ATargetPoint* Objective = World->SpawnActor<ATargetPoint>();
    Objective->SetActorLocation(Wall->GetActorLocation());
    Manager->ObjectiveTarget = Objective;
    Manager->bFollowObjective = true;
    const double OriginalDistance = FVector::Dist(Manager->GetDroneStates()[0].PositionCm, Objective->GetActorLocation());
    Manager->AdvanceOneStep();
    TestEqual(TEXT("Embedded objective accepted"), Manager->GetGroupStatuses()[0].Mode, EIstanaSwarmCommandType::FollowWaypoints);
    bool bObjectiveClear = true;
    for (int32 Step = 0; Step < 400; ++Step)
    {
        const FVector Before = Manager->GetDroneStates()[0].PositionCm;
        Manager->AdvanceOneStep();
        FHitResult Hit;
        bObjectiveClear &= !World->SweepSingleByChannel(Hit, Before, Manager->GetDroneStates()[0].PositionCm,
            FQuat::Identity, ECC_Visibility, FCollisionShape::MakeSphere(20));
    }
    TestTrue(TEXT("Approaches embedded marker"), FVector::Dist(Manager->GetDroneStates()[0].PositionCm, Objective->GetActorLocation()) < OriginalDistance - 300);
    TestTrue(TEXT("Partial approach retains collision"), bObjectiveClear);
    TestTrue(TEXT("Partial status exposed"), Manager->GetGroupStatuses()[0].bHasPartialPath);
    TestFalse(TEXT("Partial approach does not claim exact arrival"), Manager->GetGroupStatuses()[0].bRouteCompleted);
    Wall->SetActorLocation(Wall->GetActorLocation() + FVector(0, 10000, 0));
    World->Tick(LEVELTICK_All, 0.05f);
    for (int32 Step = 0; Step < 400; ++Step) Manager->AdvanceOneStep();
    TestTrue(TEXT("Unchanged objective retried after obstruction moves"), Manager->GetGroupStatuses()[0].bRouteCompleted);
    TestTrue(TEXT("Reaches exact marker after obstruction moves"), FVector::Dist(Manager->GetDroneStates()[0].PositionCm, Objective->GetActorLocation()) < 5);
    GEngine->DestroyWorldContext(World); World->DestroyWorld(false);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIstanaSwarmUnboundedTest, "Istana.Simulation.Swarm.NoArenaBoundary",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FIstanaSwarmUnboundedTest::RunTest(const FString& Parameters)
{
    FIstanaSwarmSimulation Sim;
    FString Error;
    const FVector Start(-200000, 500000, -30000);
    const FVector Goal = Start + FVector(6000, 0, -3000);
    TestTrue(TEXT("Spawn far beyond old bounds"), Sim.Initialize({MakeGroup(0, 1, Start)}, FIstanaSwarmSettings(), 1, 0.05, FGuid::NewGuid(), Error));
    TestTrue(TEXT("Distant target accepted without boundary"), Sim.SubmitCommand(Route(Sim, Goal), Error));
    for (int32 Step = 0; Step < 700; ++Step) Sim.Step();
    TestTrue(TEXT("Reaches destination outside old boundaries"), FVector::Dist(Sim.GetStates()[0].PositionCm, Goal) < 5);
    TestTrue(TEXT("Unbounded route complete"), Sim.GetGroupStatuses()[0].bRouteCompleted);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIstanaRedTeamSpawnTest, "Istana.Simulation.RedTeam.SpawnAndSharedObjective",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FIstanaRedTeamSpawnTest::RunTest(const FString& Parameters)
{
    const UWorld::InitializationValues Values = UWorld::InitializationValues()
        .AllowAudioPlayback(false).CreatePhysicsScene(true).CreateNavigation(false)
        .CreateAISystem(false).ShouldSimulatePhysics(false);
    UWorld* World = UWorld::CreateWorld(EWorldType::Game, false, NAME_None, nullptr, true, ERHIFeatureLevel::Num, &Values);
    if (!TestNotNull(TEXT("Test world"), World)) return false;
    GEngine->CreateNewWorldContext(EWorldType::Game).SetCurrentWorld(World);
    ATargetPoint* Target = World->SpawnActor<ATargetPoint>();
    Target->SetActorLocation(FVector(1000000, -2000000, 3000));
    ARedTeamManager* Manager = World->SpawnActor<ARedTeamManager>();
    Manager->bAutoInitialize = false;
    Manager->bAutoAdvance = false;
    Manager->ObjectiveTarget = Target;
    Manager->NumberOfSwarms = 3;
    Manager->DronesPerSwarm = 3;
    Manager->MinSpawnRadiusCm = 5000;
    Manager->MaxSpawnRadiusCm = 8000;
    Manager->SwarmSpreadRadiusCm = 250;
    Manager->SpawnHeightOffsetCm = 100;
    FString Error;
    if (!TestTrue(TEXT("Red team spawns"), Manager->SpawnSwarms(Error)))
    { AddError(Error); GEngine->DestroyWorldContext(World); World->DestroyWorld(false); return false; }
    TestEqual(TEXT("Configured groups"), Manager->SpawnedGroups.Num(), 3);
    TestEqual(TEXT("Configured drone count"), Manager->GetDroneStates().Num(), 9);
    for (const FIstanaSwarmConfig& Group : Manager->SpawnedGroups)
    {
        const double Radius = FVector::Dist2D(Group.SpawnOriginCm, Target->GetActorLocation());
        TestTrue(TEXT("Spawn center lies in objective annulus"), Radius >= 5000 && Radius <= 8000);
        TestEqual(TEXT("Configured height offset"), Group.SpawnOriginCm.Z, Target->GetActorLocation().Z + 100);
        for (const FIstanaDroneState& State : Manager->GetDroneStates())
            if (State.GroupId == Group.GroupId)
                TestTrue(TEXT("Member inside group spread"), FVector::Dist(State.PositionCm, Group.SpawnOriginCm) <= 250);
    }
    const TArray<FIstanaDroneState> Initial = Manager->GetDroneStates();
    TestTrue(TEXT("Reset"), Manager->ResetSimulation(Error));
    TestTrue(TEXT("Seeded reset repeats placement"), SameStates(Initial, Manager->GetDroneStates()));
    int32 Visuals = 0;
    for (TActorIterator<AIstanaDroneVisual> It(World); It; ++It)
        if (!It->IsActorBeingDestroyed() && It->GetOwner() == Manager) ++Visuals;
    TestEqual(TEXT("Visuals present without reset duplicates"), Visuals, 9);
    for (int32 Step = 0; Step < 1200; ++Step) Manager->AdvanceOneStep();
    for (const FIstanaSwarmGroupStatus& Group : Manager->GetGroupStatuses())
    {
        TestEqual(TEXT("All groups share objective"), Group.TargetCm, Target->GetActorLocation());
        TestTrue(TEXT("Groups arrive at shared objective"), Group.bRouteCompleted);
    }
    const FGuid RunBeforeMove = Manager->GetRunId();
    Target->SetActorLocation(Target->GetActorLocation() + FVector(6000, 0, 1000));
    Manager->AdvanceOneStep();
    for (const FIstanaSwarmGroupStatus& Group : Manager->GetGroupStatuses())
        TestEqual(TEXT("All groups retarget"), Group.TargetCm, Target->GetActorLocation());
    TestEqual(TEXT("Retarget does not respawn"), Manager->GetRunId(), RunBeforeMove);
    const TArray<FIstanaDroneState> BeforeInvalid = Manager->GetDroneStates();
    Manager->NumberOfSwarms = 256;
    TestFalse(TEXT("Over-budget population rejected"), Manager->SpawnSwarms(Error));
    TestTrue(TEXT("Failed respawn retains state"), SameStates(BeforeInvalid, Manager->GetDroneStates()));
    Manager->ObjectiveTarget = nullptr;
    TestFalse(TEXT("Missing objective rejected"), Manager->SpawnSwarms(Error));
    Manager->AdvanceOneStep();
    for (const FIstanaSwarmGroupStatus& Group : Manager->GetGroupStatuses())
        TestEqual(TEXT("Clearing shared target stops all groups"), Group.Mode, EIstanaSwarmCommandType::Stop);
    Manager->Destroy();
    Visuals = 0;
    for (TActorIterator<AIstanaDroneVisual> It(World); It; ++It)
        if (!It->IsActorBeingDestroyed() && It->GetOwner() == Manager) ++Visuals;
    TestEqual(TEXT("Manager removal cleans up visuals"), Visuals, 0);
    GEngine->DestroyWorldContext(World); World->DestroyWorld(false);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIstanaSwarmGroundObjectiveTest, "Istana.Simulation.Swarm.GroundObjective",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FIstanaSwarmGroundObjectiveTest::RunTest(const FString& Parameters)
{
    FIstanaSwarmSimulation Sim;
    FString Error;
    auto Floor = [](const FVector& A, const FVector& B, double Radius)
    { return FMath::Min(A.Z, B.Z) <= Radius; };
    TestTrue(TEXT("Groups above floor"), Sim.Initialize({MakeGroup(0, 1, FVector(-1000, -600, 500)),
        MakeGroup(1, 1, FVector(-1000, 600, 500))}, FIstanaSwarmSettings(), 4, 0.05, FGuid::NewGuid(), Error, Floor));
    const FVector Goal(1000, 0, 0);
    TestFalse(TEXT("Strict route still checks collision"), Sim.SubmitCommand(Route(Sim, Goal), Error));
    for (int32 Group = 0; Group < 2; ++Group)
    {
        FIstanaSwarmCommand C = Route(Sim, Goal, Group);
        C.bAllowPartialPath = true;
        TestTrue(TEXT("Floor objective accepted for each group"), Sim.SubmitCommand(C, Error));
    }
    bool bClear = true;
    for (int32 Step = 0; Step < 500; ++Step)
    {
        Sim.Step();
        for (const FIstanaDroneState& State : Sim.GetStates()) bClear &= State.PositionCm.Z > 20;
    }
    TestTrue(TEXT("Never penetrates floor"), bClear);
    for (const FIstanaDroneState& State : Sim.GetStates())
        TestTrue(TEXT("Moves near ground-level objective"), FVector::Dist(State.PositionCm, Goal) < 600);
    for (const FIstanaSwarmGroupStatus& Group : Sim.GetGroupStatuses())
    {
        TestEqual(TEXT("Original objective retained"), Group.TargetCm, Goal);
        TestTrue(TEXT("Approach status"), Group.bHasPartialPath);
        TestFalse(TEXT("No false exact arrival"), Group.bRouteCompleted);
    }
    return true;
}
#endif
