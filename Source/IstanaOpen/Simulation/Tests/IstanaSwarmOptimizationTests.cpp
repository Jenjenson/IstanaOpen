#include "Simulation/Swarm/IstanaSwarmSimulation.h"
#include "Simulation/Swarm/IstanaSwarmSpatialIndex.h"
#include "Simulation/Tests/Reference/IstanaReferenceSwarmSimulation.h"
#include "Simulation/RedTeam/RedTeamManager.h"
#include "Misc/AutomationTest.h"
#include "JsonObjectConverter.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformTime.h"
#include "Engine/World.h"
#include "Engine/Engine.h"
#include "Engine/TargetPoint.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"

#if WITH_DEV_AUTOMATION_TESTS
namespace
{
    template<class T> bool ExactStruct(const T& A, const T& B)
    { return T::StaticStruct()->CompareScriptStruct(&A, &B, 0); }
    template<class T> bool ExactArray(const TArray<T>& A, const TArray<T>& B)
    {
        if (A.Num() != B.Num()) return false;
        for (int32 I = 0; I < A.Num(); ++I) if (!ExactStruct(A[I], B[I])) return false;
        return true;
    }
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSwarmExactReplay, "Istana.Simulation.Optimization.ExactReferenceReplay",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSwarmExactReplay::RunTest(const FString& Parameters)
{
    FString Csv = TEXT("drones,groups,seed,scene,reference_p50_ms,optimized_p50_ms,reference_p95_ms,optimized_p95_ms,reference_p99_ms,optimized_p99_ms,reference_max_ms,optimized_max_ms\n");
    for (int32 Count : {12, 36, 60, 96, 120, 128, 256})
    for (int32 Seed : {7, 41, 12345})
    for (int32 Scene = 0; Scene < 2; ++Scene)
    {
        FIstanaSwarmSettings Settings; Settings.MaxDrones = 256;
        // Cap only this test's shared old/new search budget to keep regression runtime bounded.
        Settings.MaxNavigationNodes = 300;
        TArray<FIstanaSwarmConfig> Configs;
        for (int32 I = 0; I < Count; I += 12)
        {
            FIstanaSwarmConfig C; C.GroupId = Configs.Num(); C.DroneCount = FMath::Min(12, Count - I);
            C.SpawnOriginCm = FVector(-3000, C.GroupId * 1200 - 12000, 1200); C.SpawnRadiusCm = 400;
            C.MovementPresetId = TEXT("Boids"); Configs.Add(C);
        }
        bool bFloor = Scene == 1;
        auto Collision = [&bFloor](const FVector& A, const FVector& B, double R) { return bFloor && FMath::Min(A.Z, B.Z) <= R; };
        FIstanaSwarmSimulation New; FIstanaReferenceSwarmSimulation Old;
        FString Error; const FGuid Run(1, 2, 3, 4);
        if (!TestTrue(TEXT("New spawn"), New.Initialize(Configs, Settings, Seed, .05, Run, Error, Collision))
            || !TestTrue(TEXT("Reference spawn"), Old.Initialize(Configs, Settings, Seed, .05, Run, Error, Collision))) return false;
        TArray<double> OldTimes, NewTimes;
        for (int32 Step = 0; Step < 220; ++Step)
        {
            // Moving/removing geometry and retargets occur at identical logical boundaries.
            if (Step == 160) bFloor = false;
            if (Step == 0 || Step == 120 || Step == 180)
                for (const auto& C : Configs)
                {
                    FIstanaSwarmCommand Command; Command.RunId = Run; Command.DecisionStep = Step;
                    Command.SequenceNumber = Step; Command.GroupId = C.GroupId; Command.bAllowPartialPath = true;
                    Command.Type = EIstanaSwarmCommandType::FollowWaypoints;
                    Command.WaypointsCm.Add(FVector(1000 + Step, C.SpawnOriginCm.Y, Scene ? 0 : 1200));
                    TestEqual(TEXT("Command acceptance"), New.SubmitCommand(Command, Error), Old.SubmitCommand(Command, Error));
                }
            double T = FPlatformTime::Seconds(); Old.Step(); OldTimes.Add((FPlatformTime::Seconds() - T) * 1000);
            T = FPlatformTime::Seconds(); New.Step(); NewTimes.Add((FPlatformTime::Seconds() - T) * 1000);
            if (!ExactArray(New.GetStates(), Old.GetStates()) || !ExactArray(New.GetGroupStatuses(), Old.GetGroupStatuses())
                || !ExactStruct(New.GetDiagnostics(), Old.GetDiagnostics())
                || New.GetNavigationFingerprint() != Old.GetNavigationFingerprint())
            {
                AddError(FString::Printf(TEXT("Exact replay diverged count=%d seed=%d scene=%d step=%d states=%d groups=%d diagnostics=%d"), Count, Seed, Scene, Step,
                    ExactArray(New.GetStates(), Old.GetStates()), ExactArray(New.GetGroupStatuses(), Old.GetGroupStatuses()), ExactStruct(New.GetDiagnostics(), Old.GetDiagnostics())));
                FString A, B;
                FJsonObjectConverter::UStructToJsonObjectString(New.GetDiagnostics(), A);
                FJsonObjectConverter::UStructToJsonObjectString(Old.GetDiagnostics(), B);
                AddInfo(A); AddInfo(B);
                for (int32 I = 0; I < Count; ++I) if (!ExactStruct(New.GetStates()[I], Old.GetStates()[I]))
                { FJsonObjectConverter::UStructToJsonObjectString(New.GetStates()[I], A); FJsonObjectConverter::UStructToJsonObjectString(Old.GetStates()[I], B); AddInfo(A); AddInfo(B); break; }
                return false;
            }
        }
        OldTimes.Sort(); NewTimes.Sort();
        Csv += FString::Printf(TEXT("%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n"), Count, Configs.Num(), Seed, Scene,
            OldTimes[110], NewTimes[110], OldTimes[209], NewTimes[209], OldTimes[217], NewTimes[217], OldTimes.Last(), NewTimes.Last());
    }
    const FString Directory = FPaths::ProjectSavedDir() / TEXT("Benchmarks");
    IFileManager::Get().MakeDirectory(*Directory, true);
    TestTrue(TEXT("Write benchmark"), FFileHelper::SaveStringToFile(Csv, *(Directory / TEXT("swarm-reference.csv"))));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSwarmSpatialParity, "Istana.Simulation.Optimization.SpatialBroadPhase",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSwarmSpatialParity::RunTest(const FString& Parameters)
{
    for (double Scale : {0.01, 1.0, 10000.0, 1.e18})
    {
        FRandomStream Random(76); TArray<FIstanaDroneState> States;
        for (int32 I = 0; I < 256; ++I)
        { FIstanaDroneState S; S.PositionCm = Random.VRand() * Scale; States.Add(S); }
        FIstanaSwarmSpatialIndex Index; Index.Build(States, 140);
        for (int32 A = 0; A < States.Num(); ++A)
        {
            TArray<int32> Candidates; Index.Candidates(States[A].PositionCm, Candidates);
            for (int32 B = 0; B < States.Num(); ++B)
                if (FVector::Dist(States[A].PositionCm, States[B].PositionCm) < 140)
                    TestTrue(TEXT("Broad phase contains every exact neighbor"), Candidates.Contains(B));
            for (int32 I = 1; I < Candidates.Num(); ++I)
                TestTrue(TEXT("Stable, unique ordering"), Candidates[I - 1] < Candidates[I]);
        }
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRedTeamPlacementTest, "Istana.Simulation.RedTeam.AgentPlacement",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FRedTeamPlacementTest::RunTest(const FString& Parameters)
{
    const auto Values = UWorld::InitializationValues().AllowAudioPlayback(false).CreatePhysicsScene(true)
        .CreateNavigation(false).CreateAISystem(false).ShouldSimulatePhysics(false);
    UWorld* World = UWorld::CreateWorld(EWorldType::Game, false, NAME_None, nullptr, true, ERHIFeatureLevel::Num, &Values);
    GEngine->CreateNewWorldContext(EWorldType::Game).SetCurrentWorld(World);
    auto* Manager = World->SpawnActor<ARedTeamManager>(); Manager->bAutoInitialize = false; Manager->bSpawnVisuals = false;
    Manager->PlacementSource = ERedTeamPlacementSource::AgentPlacement;
    Manager->ObjectiveTarget = World->SpawnActor<ATargetPoint>(); Manager->NumberOfSwarms = 2; Manager->DronesPerSwarm = 3;
    FString Error; FRedTeamPlacementContext Context;
    TestTrue(TEXT("Begin"), Manager->BeginPlacementEpisode(42, Context, Error));
    TestEqual(TEXT("No random fallback"), Manager->GetDroneStates().Num(), 0);
    FRedTeamPlacementAction Action; Action.RunId = Context.RunId; Action.Revision = Context.Revision;
    for (int32 I = 0; I < 2; ++I)
    { FRedTeamPlacementCenter C; C.GroupId = I; C.CenterWorldCm = FVector(I ? 4000 : -4000, 0, 0); Action.Centers.Add(C); }
    TestTrue(TEXT("Validation"), Manager->ValidatePlacement(Action, Error));
    TestEqual(TEXT("Validation has no spawn side effect"), Manager->GetDroneStates().Num(), 0);
    auto Result = Manager->SubmitPlacement(Action);
    TestTrue(TEXT("Commit"), Result.bAccepted);
    TestEqual(TEXT("Exact agent center"), Manager->SpawnedGroups[0].SpawnOriginCm, Action.Centers[0].CenterWorldCm);
    Manager->AdvanceOneStep(); TestEqual(TEXT("Base clock cannot advance agent"), Manager->GetDiagnostics().ExecutedSteps, int64(0));
    FRedTeamEpisodeObservation Observation;
    TestTrue(TEXT("Step"), Manager->AdvanceEpisode(11, Observation, Error));
    TestEqual(TEXT("Exactly requested steps"), Observation.CompletedSteps, int64(11));
    TestEqual(TEXT("No dropped wall time"), Manager->DroppedWallSeconds, 0.0);
    TestTrue(TEXT("Retry returns original result"), Manager->SubmitPlacement(Action).bAccepted);
    TestEqual(TEXT("Retry does not respawn"), Manager->GetDiagnostics().ExecutedSteps, int64(11));
    Action.Centers[0].CenterWorldCm.X -= 1;
    TestFalse(TEXT("Conflicting retry rejected"), Manager->SubmitPlacement(Action).bAccepted);
    ++Action.RequestId; TestFalse(TEXT("Mid-flight relocation rejected"), Manager->SubmitPlacement(Action).bAccepted);
    const auto Previous = Manager->GetDroneStates();
    TestTrue(TEXT("New episode"), Manager->BeginPlacementEpisode(42, Context, Error));
    TestTrue(TEXT("Retained previous run"), ExactArray(Previous, Manager->GetDroneStates()));
    TestEqual(TEXT("Old state isolated from new observation"), Manager->GetEpisodeObservation().Drones.Num(), 0);
    TestFalse(TEXT("Stale action"), Manager->SubmitPlacement(Action).bAccepted);
    Action.RunId = Context.RunId; Action.Revision = Context.Revision; Action.RequestId = 0;
    Action.Centers[0].CenterWorldCm.X = -4000; Action.Centers[1].GroupId = 0;
    TestFalse(TEXT("Duplicate group"), Manager->SubmitPlacement(Action).bAccepted);
    Action.Centers[1].GroupId = 1; ++Action.RequestId;
    Manager->ObjectiveTarget->SetActorLocation(FVector(10, 0, 0));
    TestFalse(TEXT("Changed objective"), Manager->SubmitPlacement(Action).bAccepted);
    Manager->ObjectiveTarget->SetActorLocation(FVector::ZeroVector);
    Manager->NotifyPlacementWorldChanged(); ++Action.RequestId;
    TestFalse(TEXT("Changed world context"), Manager->SubmitPlacement(Action).bAccepted);
    Manager->CancelEpisode(TEXT("disconnect"));
    TestTrue(TEXT("Explicit truncation"), Manager->GetEpisodeObservation().bTruncated);
    TestFalse(TEXT("Cancelled cannot step"), Manager->AdvanceEpisode(1, Observation, Error));

    // A blocked member region cannot trigger a different center or damage the retained run.
    auto* Blocker = World->SpawnActor<AActor>();
    auto* Mesh = NewObject<UStaticMeshComponent>(Blocker); Blocker->SetRootComponent(Mesh);
    Mesh->SetStaticMesh(LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube")));
    Mesh->SetCollisionProfileName(TEXT("BlockAll")); Mesh->RegisterComponent();
    Blocker->SetActorLocation(FVector(-4000,0,0)); Blocker->SetActorScale3D(FVector(20));
    World->Tick(LEVELTICK_All,.05f);
    // A simple box provides a solid volume; triangle collision represents its surface.
    Manager->bTraceComplexObstacles=false;
    TestTrue(TEXT("Begin blocked placement episode"), Manager->BeginPlacementEpisode(42, Context, Error));
    Action.RunId=Context.RunId; Action.Revision=Context.Revision; Action.RequestId=0;
    const auto BeforeBlocked=Manager->GetDroneStates();
    TestFalse(TEXT("Blocked member region rejects placement"), Manager->SubmitPlacement(Action).bAccepted);
    TestTrue(TEXT("Blocked placement preserves old state"), ExactArray(BeforeBlocked,Manager->GetDroneStates()));
    Mesh->SetCollisionEnabled(ECollisionEnabled::NoCollision); World->Tick(LEVELTICK_All,.05f);
    // A rejected ID remains rejected on retry; a new ID validates fresh world collision.
    TestFalse(TEXT("Rejected action retry remains idempotent"), Manager->SubmitPlacement(Action).bAccepted);
    ++Action.RequestId;
    TestTrue(TEXT("Fresh action sees removed collision"), Manager->SubmitPlacement(Action).bAccepted);
    Manager->RestartDemo();
    TestEqual(TEXT("Agent restart awaits placement"), Manager->EpisodePhase, ERedTeamEpisodePhase::AwaitingPlacement);
    TestEqual(TEXT("Agent restart exposes no retained state"), Manager->GetEpisodeObservation().Drones.Num(),0);
    auto* Provider=NewObject<URedTeamScriptedPlacementPolicy>(Manager);
    Provider->Manager=Manager; Provider->Centers=Action.Centers;
    Manager->PlacementPolicy.SetObject(Provider); Manager->PlacementPolicy.SetInterface(Provider);
    TestTrue(TEXT("Local provider episode"),Manager->BeginPlacementEpisode(42,Context,Error));
    TestEqual(TEXT("Provider supplied placement"),Manager->EpisodePhase,ERedTeamEpisodePhase::Running);
    TestEqual(TEXT("Provider center honored"),Manager->SpawnedGroups[0].SpawnOriginCm,Provider->Centers[0].CenterWorldCm);
    GEngine->DestroyWorldContext(World); World->DestroyWorld(false);
    return true;
}
#endif
