#include "Simulation/RedTeam/RedTeamAgentBridge.h"
#include "Simulation/RedTeam/RedTeamManager.h"
#include "Simulation/BlueTeam/BlueTeamCoordinator.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/AutomationTest.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "Engine/World.h"
#include "Engine/Engine.h"
#include "Engine/TargetPoint.h"
#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBlueTrainingConfiguration, "Istana.Simulation.BlueTeam.TrainingConfiguration",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FBlueTrainingConfiguration::RunTest(const FString& Parameters)
{
    const auto Values = UWorld::InitializationValues().AllowAudioPlayback(false).CreatePhysicsScene(true)
        .CreateNavigation(false).CreateAISystem(false).ShouldSimulatePhysics(false);
    UWorld* World = UWorld::CreateWorld(EWorldType::Game, false, NAME_None, nullptr, true, ERHIFeatureLevel::Num, &Values);
    GEngine->CreateNewWorldContext(EWorldType::Game).SetCurrentWorld(World);
    auto* Manager = World->SpawnActor<ARedTeamManager>();
    Manager->bAutoInitialize = false; Manager->bSpawnVisuals = false;
    Manager->PlacementSource = ERedTeamPlacementSource::AgentPlacement;
    Manager->ObjectiveTarget = World->SpawnActor<ATargetPoint>();
    auto* Blue = World->SpawnActor<ABlueTeamCoordinator>();
    Blue->Manager = Manager; Manager->BlueCoordinator = Blue;
    Blue->bSpawnSensorMarkers = false; Blue->bDrawCoverage = false;
    auto* Bridge = World->SpawnActor<ARedTeamAgentBridge>(); Bridge->Manager = Manager;
    int32 RequestId = 0;
    auto Reset = [&](const FString& Fields)
    {
        const FString Request = FString::Printf(TEXT("{\"id\":%d,\"op\":\"reset\",\"seed\":41%s}"), RequestId++, *Fields);
        TSharedPtr<FJsonObject> Response;
        const bool bDecoded = FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Bridge->HandleRequest(Request)), Response);
        TestTrue(TEXT("Reset response decodes"), bDecoded && Response.IsValid());
        return Response;
    };
    TestTrue(TEXT("Ordinary reset remains supported"), Reset(TEXT(""))->GetBoolField(TEXT("ok")));
    TestTrue(TEXT("Training reset accepts native budget and sensor selection"),
        Reset(TEXT(",\"blueConfiguration\":{\"budget\":10,\"availableSensorIds\":[\"thermal\"]}"))->GetBoolField(TEXT("ok")));
    const auto Context = Blue->ContextJson();
    TestEqual(TEXT("Capability is explicit"), Context->GetNumberField(TEXT("trainingConfigurationVersion")), 1.);
    TestTrue(TEXT("Existing first-detection warning definition is preserved"), Context->GetStringField(TEXT("warningDefinition")).Contains(TEXT("first detection")));
    const auto Snapshot = Context->GetObjectField(TEXT("publicSnapshot"));
    TestEqual(TEXT("Actual total budget is advertised"), Snapshot->GetNumberField(TEXT("budget_total")), 10.);
    TestEqual(TEXT("Actual remaining budget is advertised"), Snapshot->GetNumberField(TEXT("budget_remaining")), 10.);
    TestEqual(TEXT("Catalogue retains all genuine sensor profiles"), Context->GetArrayField(TEXT("catalogue")).Num(), Blue->Catalogue.Num());
    TestEqual(TEXT("Native availability is configured"), Blue->AvailableSensorIds.Num(), 1);
    TestEqual(TEXT("Directional thermal remains enabled"), Blue->AvailableSensorIds[0], FString(TEXT("thermal")));
    const FGuid AcceptedRun = Manager->GetPlacementContext().RunId;
    for (const FString Bad : {
        TEXT("null"), TEXT("{\"budget\":-1,\"availableSensorIds\":[]}"),
        TEXT("{\"budget\":10,\"availableSensorIds\":[\"missing\"]}"),
        TEXT("{\"budget\":10,\"availableSensorIds\":[\"thermal\",\"thermal\"]}"),
        TEXT("{\"budget\":10,\"availableSensorIds\":[1]}"),
        TEXT("{\"budget\":10,\"availableSensorIds\":[],\"extra\":true}"),
        TEXT("{\"budget\":10}")})
    {
        TestFalse(TEXT("Malformed configuration is rejected"), Reset(TEXT(",\"blueConfiguration\":") + Bad)->GetBoolField(TEXT("ok")));
        TestEqual(TEXT("Rejected reset preserves episode"), Manager->GetPlacementContext().RunId, AcceptedRun);
        TestEqual(TEXT("Rejected reset preserves budget"), Blue->Budget, 10.);
        TestEqual(TEXT("Rejected reset preserves availability"), Blue->AvailableSensorIds.Num(), 1);
    }
    const int32 Swarms = Manager->NumberOfSwarms; Manager->NumberOfSwarms = 0;
    TestFalse(TEXT("Unrelated environment validation can reject reset"),
        Reset(TEXT(",\"blueConfiguration\":{\"budget\":5,\"availableSensorIds\":[\"rf\"]}"))->GetBoolField(TEXT("ok")));
    TestEqual(TEXT("Failed environment reset rolls budget back"), Blue->Budget, 10.);
    TestEqual(TEXT("Failed environment reset rolls availability back"), Blue->AvailableSensorIds[0], FString(TEXT("thermal")));
    Manager->NumberOfSwarms = Swarms;
    TestTrue(TEXT("Ordinary reset preserves current configuration"), Reset(TEXT(""))->GetBoolField(TEXT("ok")));
    TestEqual(TEXT("Configured budget survives ordinary reset"), Blue->Budget, 10.);
    TestTrue(TEXT("All-disabled sensor control is legal"),
        Reset(TEXT(",\"blueConfiguration\":{\"budget\":2,\"availableSensorIds\":[]}"))->GetBoolField(TEXT("ok")));
    TestEqual(TEXT("All sensor types are disabled natively"), Blue->AvailableSensorIds.Num(), 0);
    GEngine->DestroyWorldContext(World); World->DestroyWorld(false);
    return true;
}
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
