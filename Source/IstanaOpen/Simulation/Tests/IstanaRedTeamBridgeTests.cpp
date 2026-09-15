#include "Simulation/RedTeam/RedTeamAgentBridge.h"
#include "Simulation/RedTeam/RedTeamManager.h"
#include "Misc/AutomationTest.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "Engine/World.h"
#include "Engine/Engine.h"
#include "Engine/TargetPoint.h"
#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRedTeamBridgeRoundTrip, "Istana.Simulation.RedTeam.ExternalPythonRoundTrip",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FRedTeamBridgeRoundTrip::RunTest(const FString& Parameters)
{
    const auto Values=UWorld::InitializationValues().AllowAudioPlayback(false).CreatePhysicsScene(true)
        .CreateNavigation(false).CreateAISystem(false).ShouldSimulatePhysics(false);
    UWorld* World=UWorld::CreateWorld(EWorldType::Game,false,NAME_None,nullptr,true,ERHIFeatureLevel::Num,&Values);
    GEngine->CreateNewWorldContext(EWorldType::Game).SetCurrentWorld(World);
    auto* Manager=World->SpawnActor<ARedTeamManager>(); Manager->bAutoInitialize=false; Manager->bSpawnVisuals=false;
    Manager->PlacementSource=ERedTeamPlacementSource::AgentPlacement; Manager->NumberOfSwarms=2; Manager->DronesPerSwarm=3;
    Manager->ObjectiveTarget=World->SpawnActor<ATargetPoint>();
    auto* Bridge=World->SpawnActor<ARedTeamAgentBridge>(); Bridge->Manager=Manager; Bridge->Port=18765; Bridge->IdleTimeoutSeconds=1;
    FString Error; bool bSuccess=TestTrue(TEXT("Start bridge"),Bridge->StartBridge(Error));
    if (bSuccess)
    {
        FString Python=FPaths::ConvertRelativePathToFull(FPaths::EngineDir()/TEXT("Binaries/ThirdParty/Python3/Win64/python.exe"));
        FString Script=FPaths::ConvertRelativePathToFull(FPaths::ProjectDir()/TEXT("Tools/test_red_team_runtime.py"));
        FString Args=FString::Printf(TEXT("\"%s\" %d"),*Script,Bridge->Port);
        auto Process=FPlatformProcess::CreateProc(*Python,*Args,false,true,true,nullptr,0,nullptr,nullptr);
        bSuccess=TestTrue(TEXT("Launch external Python client"),Process.IsValid());
        if (Process.IsValid())
        {
            const double Deadline=FPlatformTime::Seconds()+30;
            while(FPlatformProcess::IsProcRunning(Process) && FPlatformTime::Seconds()<Deadline)
            { Bridge->Tick(.01f); FPlatformProcess::Sleep(.005f); }
            if (FPlatformProcess::IsProcRunning(Process)) { FPlatformProcess::TerminateProc(Process); AddError(TEXT("External client timed out")); bSuccess=false; }
            else { int32 Code=-1; FPlatformProcess::GetProcReturnCode(Process,&Code); bSuccess=TestEqual(TEXT("External Python assertions passed"),Code,0); }
            FPlatformProcess::CloseProc(Process);
        }
        Bridge->StopBridge();
        TestTrue(TEXT("Disconnect leaves episode frozen"),Manager->GetEpisodeObservation().bTruncated);
        const int64 Step=Manager->GetDiagnostics().ExecutedSteps;
        Manager->Tick(1); TestEqual(TEXT("No automatic step after disconnect"),Manager->GetDiagnostics().ExecutedSteps,Step);
    }
    GEngine->DestroyWorldContext(World); World->DestroyWorld(false);
    return bSuccess;
}
#endif
