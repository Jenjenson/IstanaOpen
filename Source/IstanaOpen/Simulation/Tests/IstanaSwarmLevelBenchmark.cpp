#include "CoreMinimal.h"
#if WITH_DEV_AUTOMATION_TESTS && WITH_EDITOR
#include "Simulation/Tests/Reference/IstanaReferenceSwarmSimulation.h"
#include "Simulation/RedTeam/RedTeamManager.h"
#include "Engine/TargetPoint.h"
#include "EngineUtils.h"
#include "Engine/World.h"
#include "Engine/StaticMesh.h"
#include "Components/StaticMeshComponent.h"
#include "FileHelpers.h"
#include "Misc/AutomationTest.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformTime.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSwarmLevelBenchmark, "Istana.Performance.Swarm.LevelCollision",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FSwarmLevelBenchmark::RunTest(const FString& Parameters)
{
    // Load in a separate unattended editor process. Never save the level or its actors.
    UWorld* World = UEditorLoadingAndSavingUtils::LoadMap(FPaths::ProjectContentDir() / TEXT("Maps/Istana.umap"));
    if (!TestNotNull(TEXT("Istana map"), World)) return false;
    World->UpdateWorldComponents(false, false);
    ARedTeamManager* Template = nullptr;
    for (TActorIterator<ARedTeamManager> It(World); It; ++It) { Template = *It; break; }
    if (!TestNotNull(TEXT("Map red team manager configuration"), Template) || !TestNotNull(TEXT("Map objective"), Template->ObjectiveTarget.Get())) return false;
    FString Csv = TEXT("drones,groups,seed,spawn_ms,command_reference_ms,command_new_ms,old_p50_ms,new_p50_ms,old_p95_ms,new_p95_ms,old_p99_ms,new_p99_ms,old_max_ms,new_max_ms,queries,expanded_nodes\n");
    TSet<FString> CompletedCases;
    if (FParse::Param(FCommandLine::Get(), TEXT("SwarmBenchmarkResume")))
    {
        FString Previous;
        if (FFileHelper::LoadFileToString(Previous, *(FPaths::ProjectSavedDir()/TEXT("Benchmarks/swarm-level.csv"))))
        {
            if (!Previous.StartsWith(Csv)) { AddError(TEXT("Cannot resume: benchmark CSV schema differs.")); return false; }
            Csv = Previous;
            TArray<FString> Lines; Previous.ParseIntoArrayLines(Lines);
            for (int32 I=1; I<Lines.Num(); ++I)
            {
                TArray<FString> Fields; Lines[I].ParseIntoArray(Fields,TEXT(","));
                if (Fields.Num()==16) CompletedCases.Add(Fields[0]+TEXT(":")+Fields[2]);
            }
        }
    }
    for (int32 Count : {60,12,36,96,120,128,256})
    for (int32 Seed : {7,41,12345})
    {
        if (CompletedCases.Contains(FString::Printf(TEXT("%d:%d"),Count,Seed))) continue;
        UE_LOG(LogTemp, Display, TEXT("Swarm benchmark starting %d drones seed %d"),Count,Seed);
        FActorSpawnParameters Params; Params.Template = Template;
        auto* Manager = World->SpawnActor<ARedTeamManager>(Template->GetActorLocation(), Template->GetActorRotation(), Params);
        Manager->bAutoInitialize = false; Manager->bAutoAdvance = false; Manager->bSpawnVisuals = false; Manager->bDrawDebug = false;
        Manager->PlacementSource = ERedTeamPlacementSource::SeededLayout;
        Manager->MovementPreset = nullptr; Manager->Settings = Template->MovementPreset ? Template->MovementPreset->Settings : Template->Settings;
        Manager->Settings.MaxDrones = 256; Manager->NumberOfSwarms = FMath::DivideAndRoundUp(Count,12); Manager->DronesPerSwarm = 12;
        // Generate a feasible layout using the actual map settings; trim the last group to exact count.
        Manager->Settings.MaxDrones = 256; Manager->NumberOfSwarms = Count > 252 ? 16 : FMath::DivideAndRoundUp(Count,12);
        Manager->DronesPerSwarm = Count > 252 ? 16 : 12;
        Manager->Seed = Seed; FString Error;
        double Start = FPlatformTime::Seconds();
        if (!Manager->SpawnSwarms(Error))
        { AddWarning(FString::Printf(TEXT("Map cannot place %d drones seed %d: %s"), Count, Seed, *Error)); Manager->Destroy(); continue; }
        const double SpawnMs = (FPlatformTime::Seconds()-Start)*1000;
        auto Configs = Manager->SpawnedGroups;
        if (Count <= 252) Configs.Last().DroneCount -= Configs.Num()*12-Count;
        const FGuid Run = FGuid::NewGuid();
        if (!Manager->InitializeSimulation(Configs, Manager->Settings, Seed, Manager->FixedStepSeconds, Run, Error))
        { AddError(Error); Manager->Destroy(); return false; }
        FIstanaReferenceSwarmSimulation Reference;
        auto Query = [World, Manager](const FVector& A,const FVector& B,double Radius)
        {
            FCollisionQueryParams P(SCENE_QUERY_STAT(IstanaReference), Manager->bTraceComplexObstacles); P.AddIgnoredActor(Manager);
            const auto Shape = FCollisionShape::MakeSphere(float(Radius)); FHitResult Hit;
            return World->OverlapBlockingTestByChannel(A,FQuat::Identity,Manager->ObstacleTraceChannel,Shape,P)
                || World->SweepSingleByChannel(Hit,A,B,FQuat::Identity,Manager->ObstacleTraceChannel,Shape,P);
        };
        if (!Reference.Initialize(Configs,Manager->Settings,Seed,Manager->FixedStepSeconds,Run,Error,Query))
        { AddError(Error); Manager->Destroy(); return false; }
        Manager->bFollowObjective = false;
        double OldCommand=0,NewCommand=0;
        for (const auto& Group : Configs)
        {
            FIstanaSwarmCommand C; C.RunId=Run; C.GroupId=Group.GroupId; C.Type=EIstanaSwarmCommandType::FollowWaypoints;
            C.bAllowPartialPath=true; C.WaypointsCm.Add(Manager->ObjectiveTarget->GetActorLocation());
            Start=FPlatformTime::Seconds(); Reference.SubmitCommand(C,Error); OldCommand+=(FPlatformTime::Seconds()-Start)*1000;
            Start=FPlatformTime::Seconds(); Manager->SubmitCommand(C,Error); NewCommand+=(FPlatformTime::Seconds()-Start)*1000;
        }
        TArray<double> OldTimes,NewTimes;
        for (int32 Step=0;Step<240;++Step)
        {
            Start=FPlatformTime::Seconds(); Reference.Step(); OldTimes.Add((FPlatformTime::Seconds()-Start)*1000);
            Start=FPlatformTime::Seconds(); Manager->AdvanceOneStep(); NewTimes.Add((FPlatformTime::Seconds()-Start)*1000);
            auto States=Manager->GetDroneStates(); const auto& Expected=Reference.GetStates();
            for (int32 I=0;I<States.Num();++I)
                if (States[I].PositionCm!=Expected[I].PositionCm || States[I].VelocityCmPerSecond!=Expected[I].VelocityCmPerSecond)
                { AddError(FString::Printf(TEXT("Level replay differs count=%d seed=%d step=%d drone=%d"),Count,Seed,Step,I)); Manager->Destroy(); return false; }
        }
        OldTimes.Sort(); NewTimes.Sort(); const auto Work=Manager->GetWorkCounters();
        Csv+=FString::Printf(TEXT("%d,%d,%d,%.3f,%.3f,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%lld,%lld\n"),
            Count,Configs.Num(),Seed,SpawnMs,OldCommand,NewCommand,OldTimes[120],NewTimes[120],OldTimes[228],NewTimes[228],
            OldTimes[237],NewTimes[237],OldTimes.Last(),NewTimes.Last(),Work.CollisionQueries,Work.ExpandedNodes);
        AddInfo(FString::Printf(TEXT("Completed actual-map replay %d drones seed %d"),Count,Seed));
        Manager->Destroy();
        const FString Directory=FPaths::ProjectSavedDir()/TEXT("Benchmarks"); IFileManager::Get().MakeDirectory(*Directory,true);
        FFileHelper::SaveStringToFile(Csv,*(Directory/TEXT("swarm-level.csv")));
    }
    return true;
}
#endif
