#include "Simulation/BlueTeam/BlueTeamCoordinator.h"
#include "Simulation/BlueTeam/BlueWarningTime.h"
#include "Simulation/RedTeam/RedTeamManager.h"
#include "IstanaGameMode.h"
#include "Dom/JsonObject.h"
#include "Engine/Engine.h"
#include "Engine/TargetPoint.h"
#include "Engine/World.h"
#include "Engine/StaticMeshActor.h"
#include "Components/StaticMeshComponent.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBlueWarningTimeContract, "Istana.Simulation.BlueTeam.WarningTime",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FBlueWarningTimeContract::RunTest(const FString& Parameters)
{
    TestEqual(TEXT("Interpolated arrival minus first detection"), BlueWarningSeconds(2., 20.75), 18.75);
    TestEqual(TEXT("Undetected arrival has zero warning"), BlueWarningSeconds(-1., 20.), 0.);
    TestEqual(TEXT("Late detection cannot give negative warning"), BlueWarningSeconds(21., 20.), 0.);
    TestEqual(TEXT("Initial zone entry has zero warning"), BlueWarningSeconds(0., 0.), 0.);
    TestEqual(TEXT("Unresolved target contributes zero lower bound"), BlueWarningSeconds(1., -1.), 0.);
    return true;
}
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBlueLiveApproachModes, "Istana.Simulation.BlueTeam.LiveApproachModes",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FBlueLiveApproachModes::RunTest(const FString& Parameters)
{
    const auto Values = UWorld::InitializationValues().AllowAudioPlayback(false).CreatePhysicsScene(true)
        .CreateNavigation(false).CreateAISystem(false).ShouldSimulatePhysics(false);
    UWorld* World = UWorld::CreateWorld(EWorldType::Game, false, NAME_None, nullptr, true, ERHIFeatureLevel::Num, &Values);
    GEngine->CreateNewWorldContext(EWorldType::Game).SetCurrentWorld(World);
    ARedTeamManager* Manager = World->SpawnActor<ARedTeamManager>();
    ABlueTeamCoordinator* Blue = World->SpawnActor<ABlueTeamCoordinator>();
    Manager->MinSpawnRadiusCm = 3000; Manager->MaxSpawnRadiusCm = 10000;
    Manager->SwarmSpreadRadiusCm = 1000; Manager->SpawnHeightOffsetCm = 0;

    AIstanaGameMode::ConfigureBlueLiveApproach(*Manager, *Blue, false, false, false);
    TestEqual(TEXT("Ordinary live mode preserves the map minimum"), Manager->MinSpawnRadiusCm, 3000.);
    TestEqual(TEXT("Ordinary live mode preserves the map maximum"), Manager->MaxSpawnRadiusCm, 10000.);

    AIstanaGameMode::ConfigureBlueLiveApproach(*Manager, *Blue, false, true, false);
    TestEqual(TEXT("Delayed demo minimum"), Manager->MinSpawnRadiusCm, 56000.);
    TestEqual(TEXT("Delayed demo maximum"), Manager->MaxSpawnRadiusCm, 58000.);
    TestEqual(TEXT("Delayed demo preserves height"), Manager->SpawnHeightOffsetCm, 0.);
    TestEqual(TEXT("Delayed demo advertises its approach radius"), Blue->PriorSpawnRadiusM, 570.);
    TestEqual(TEXT("Delayed demo advertises its physical altitude"), Blue->PriorAltitudeM, 0.);
    TestEqual(TEXT("Delayed demo advertises its physical speed"), Blue->PriorSpeedMps,
        Manager->Settings.CruiseSpeedCmPerSecond / 100.);
    TestEqual(TEXT("Delayed demo allows physical approach time"), Blue->TimeLimitSeconds, 220.);
    double MaximumSiteRadiusM = 0, MaximumSensorRangeM = 0;
    for (const FVector2D& Site : Blue->ApprovedSitesM) MaximumSiteRadiusM = FMath::Max(MaximumSiteRadiusM, Site.Size());
    for (const FBlueSensorProfile& Profile : Blue->Catalogue)
        for (int32 I = 0; I < 4; ++I) MaximumSensorRangeM = FMath::Max(MaximumSensorRangeM, double(Profile.RangesM[I]));
    const double MinimumMemberRadiusM = Manager->MinSpawnRadiusCm / 100. - Manager->SwarmSpreadRadiusCm / 100.;
    TestTrue(TEXT("Every allowed initial drone is outside every possible sensor range"),
        MinimumMemberRadiusM - MaximumSiteRadiusM > MaximumSensorRangeM);
    const double ScriptedStartRadiusM = (Manager->MinSpawnRadiusCm + Manager->MaxSpawnRadiusCm) / 200.;
    const double CruiseSpeedMps = Manager->Settings.CruiseSpeedCmPerSecond / 100.;
    TestTrue(TEXT("Nominal inward approach reaches the objective within the frozen horizon"),
        (ScriptedStartRadiusM - Blue->ObjectiveRadiusM) / CruiseSpeedMps < Blue->TimeLimitSeconds);

    AIstanaGameMode::ConfigureBlueLiveApproach(*Manager, *Blue, true, true, false);
    TestEqual(TEXT("Warning benchmark takes precedence"), Manager->MinSpawnRadiusCm, 26000.);
    TestEqual(TEXT("Warning benchmark maximum is unchanged"), Manager->MaxSpawnRadiusCm, 30000.);
    TestEqual(TEXT("Warning benchmark height is unchanged"), Manager->SpawnHeightOffsetCm, 12000.);
    TestEqual(TEXT("Warning benchmark horizon is unchanged"), Blue->TimeLimitSeconds, 180.);
    TestEqual(TEXT("Warning benchmark prior is unchanged"), Blue->PriorSpawnRadiusM, 280.);
    Blue->Budget = 3; Blue->MaxSites = 3; Blue->TimeLimitSeconds = 96; Manager->DronesPerSwarm = 12;
    AIstanaGameMode::ConfigureBlueLiveApproach(*Manager, *Blue, false, true, true);
    TestEqual(TEXT("Training workbench provides five cost units"), Blue->Budget, 5.);
    TestEqual(TEXT("Training workbench permits five placements"), Blue->MaxSites, 5);
    TestEqual(TEXT("Training workbench uses one drone in each approach group"), Manager->DronesPerSwarm, 1);
    TestEqual(TEXT("Training public prior matches the per-group count"), Blue->PriorSwarmSize, 1);
    TestEqual(TEXT("Training lanes stay above map-specific avoidance geometry"), Manager->SpawnHeightOffsetCm, 12000.);
    TestEqual(TEXT("Training prior advertises the synthetic lane altitude"), Blue->PriorAltitudeM, 120.);
    TestEqual(TEXT("Five-lane training horizon retains complete terminal evidence"), Blue->TimeLimitSeconds, 100.);
    TestEqual(TEXT("Training camera sensing runs at a conservative five hertz"), Blue->LookIntervalSeconds, .2);
    GEngine->DestroyWorldContext(World);
    World->DestroyWorld(false);
    return true;
}
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBlueDirectionalSensorModel, "Istana.Simulation.BlueTeam.DirectionalSensorModel",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FBlueDirectionalSensorModel::RunTest(const FString& Parameters)
{
    FDirectionalSensorProfile Profile; Profile.bEnabled = true; Profile.ResolutionX = 640; Profile.ResolutionY = 512;
    Profile.HorizontalFovDegrees = 24; Profile.VerticalFovDegrees = IstanaDirectionalSensor::VerticalFovFromHorizontal(24,640,512);
    Profile.IfovMilliradians = .667; Profile.NedtMillikelvin = 20; Profile.YawBinsDegrees = {0,45}; Profile.PitchBinsDegrees = {0,10};
    TestTrue(TEXT("Calculated VFOV follows detector aspect ratio"), FMath::IsNearlyEqual(Profile.VerticalFovDegrees, 19.31, .05));
    const auto Near = IstanaDirectionalSensor::Evaluate(Profile, FVector::ZeroVector, FRotator::ZeroRotator,
        FVector(10000,0,0), .5, 1, 1, 0, 0, true);
    TestTrue(TEXT("Centred target is inside 3D frustum"), Near.bInsideFov);
    TestTrue(TEXT("0.5m target at 100m is about 7.5 pixels"), FMath::IsNearlyEqual(Near.PixelsOnTarget, 7.5, .05));
    TestTrue(TEXT("Probability is neither automatic nor zero"), Near.DetectionProbability > 0 && Near.DetectionProbability < 1);
    const auto Far = IstanaDirectionalSensor::Evaluate(Profile, FVector::ZeroVector, FRotator::ZeroRotator,
        FVector(50000,0,0), .5, 1, 1, 0, 0, true);
    TestTrue(TEXT("0.5m target at 500m is about 1.5 pixels"), FMath::IsNearlyEqual(Far.PixelsOnTarget, 1.5, .05));
    TestTrue(TEXT("Probability degrades with distance"), Far.DetectionProbability < Near.DetectionProbability);
    const auto Outside = IstanaDirectionalSensor::Evaluate(Profile, FVector::ZeroVector, FRotator::ZeroRotator,
        FRotator(0,30,0).Vector() * 10000, .5, 1, 1, 0, 0, true);
    TestFalse(TEXT("Target outside 24 degree HFOV is rejected"), Outside.bInsideFov);
    const auto Blocked = IstanaDirectionalSensor::Evaluate(Profile, FVector::ZeroVector, FRotator::ZeroRotator,
        FVector(10000,0,0), .5, 1, 1, 0, 0, false);
    TestEqual(TEXT("Blocked LOS has zero detection probability"), Blocked.DetectionProbability, 0.);
    return true;
}
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FBlueInitialLayoutContract, "Istana.Simulation.BlueTeam.InitialLayoutContract",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FBlueInitialLayoutContract::RunTest(const FString& Parameters)
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
    auto* Visual = World->SpawnActor<ABlueSensorMarker>();
    for (const FString Id : {TEXT("rf"), TEXT("radar"), TEXT("eo"), TEXT("thermal"), TEXT("fused")})
    {
        Visual->ConfigureSensor(Id, 4);
        TArray<UStaticMeshComponent*> Components; Visual->GetComponents(Components);
        TestTrue(TEXT("Detailed sensor assembly has multiple hardware parts"), Components.Num() >= 30);
        for (const auto* Component : Components)
        {
            TestEqual(TEXT("Sensor visual cannot change collision/trajectories"), Component->GetCollisionEnabled(), ECollisionEnabled::NoCollision);
            TestNotNull(TEXT("Sensor parts have authored PBR material"), Component->GetMaterial(0));
        }
        FVector Centre, Extent; Visual->GetActorBounds(false, Centre, Extent);
        TestTrue(TEXT("Sensor assembly reaches the supporting surface"), FMath::Abs(Centre.Z-Extent.Z) < .1);
    }
    Visual->Destroy();
    FRedTeamPlacementContext Context; FString Error;
    TestTrue(TEXT("Empty world reset"), Manager->BeginPlacementEpisode(122, Context, Error));
    TestEqual(TEXT("Missing surfaces block every site"), Blue->ContextJson()->GetObjectField(TEXT("publicSnapshot"))->GetArrayField(TEXT("blocked_sites")).Num(), Blue->ApprovedSitesM.Num());
    auto AddSurface = [&](FVector Location, FVector Scale)
    {
        auto* Actor = World->SpawnActor<AStaticMeshActor>(Location, FRotator::ZeroRotator);
        auto* Mesh = Actor->GetStaticMeshComponent();
        Mesh->SetStaticMesh(LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube")));
        Mesh->SetCollisionProfileName(TEXT("BlockAll"));
        Actor->SetActorScale3D(Scale);
        return Actor;
    };
    auto* Ground = AddSurface(FVector(0,0,-50), FVector(100,100,1));
    auto* Roof = AddSurface(FVector(3000,0,950), FVector(4,4,1));
    const bool bReset = Manager->BeginPlacementEpisode(123, Context, Error);
    TestTrue(TEXT("Reset with default Blue configuration"), bReset);
    if (bReset)
    {
        TestFalse(TEXT("Stepping gated before Blue commit"), Blue->CanAdvance(Error));
        auto Action = MakeShared<FJsonObject>();
        Action->SetNumberField(TEXT("schemaVersion"), 2);
        Action->SetStringField(TEXT("runId"), Context.RunId.ToString());
        Action->SetNumberField(TEXT("revision"), Context.Revision);
        Action->SetNumberField(TEXT("requestId"), 0);
        Action->SetNumberField(TEXT("expectedStep"), 0);
        Action->SetBoolField(TEXT("commit"), true);
        auto Placement = MakeShared<FJsonObject>();
        Placement->SetStringField(TEXT("profileId"), TEXT("unavailable-profile"));
        Placement->SetNumberField(TEXT("siteId"), 0);
        Placement->SetNumberField(TEXT("yawDeg"), 0);
        Placement->SetNumberField(TEXT("pitchDeg"), 0);
        TArray<TSharedPtr<FJsonValue>> Rows;
        Rows.Add(MakeShared<FJsonValueObject>(Placement));
        Action->SetArrayField(TEXT("placements"), Rows);
        TestFalse(TEXT("Unknown profile rejected atomically"), Blue->DeployJson(*Action, Error)->GetBoolField(TEXT("accepted")));
        TestFalse(TEXT("Rejected action does not commit"), Blue->IsCommitted());
        Placement->SetStringField(TEXT("profileId"), TEXT("thermal"));
        Placement->SetNumberField(TEXT("yawDeg"), 13);
        Placement->SetNumberField(TEXT("pitchDeg"), 10);
        TestFalse(TEXT("Off-bin directional orientation rejected"), Blue->DeployJson(*Action, Error)->GetBoolField(TEXT("accepted")));
        Placement->SetStringField(TEXT("profileId"), TEXT("rf"));
        Placement->SetNumberField(TEXT("yawDeg"), 0);
        Placement->SetNumberField(TEXT("pitchDeg"), 0);
        const auto Accepted = Blue->DeployJson(*Action, Error);
        TestTrue(TEXT("Valid initial sensor accepted"), Accepted->GetBoolField(TEXT("accepted")));
        TestTrue(TEXT("Layout committed"), Blue->IsCommitted());
        TestTrue(TEXT("Identical retry accepted"), Blue->DeployJson(*Action, Error)->GetBoolField(TEXT("accepted")));
        TestTrue(TEXT("Configuration remains stable after deploy"), Blue->CanAdvance(Error));
        const auto Snapshot = Blue->ContextJson()->GetObjectField(TEXT("publicSnapshot"));
        TestEqual(TEXT("No private Red tracks before sensing"), Snapshot->GetArrayField(TEXT("tracks")).Num(), 0);
        TestEqual(TEXT("One accepted placement"), Snapshot->GetArrayField(TEXT("placements")).Num(), 1);
        TestTrue(TEXT("Sensor head is mast height above roof, not above objective"),
            Blue->SiteWorldCm(0, Blue->Catalogue[0]).Equals(FVector(3000, 0, 1400), .1));
        TestTrue(TEXT("Surface coordinates are public"),
            FMath::IsNearlyEqual(Blue->ContextJson()->GetArrayField(TEXT("siteSurfacesWorldCm"))[0]->AsArray()[2]->AsNumber(), 1000., .1));
        Placement->SetNumberField(TEXT("siteId"), 1);
        TestFalse(TEXT("Conflicting retry rejected"), Blue->DeployJson(*Action, Error)->GetBoolField(TEXT("accepted")));
        Roof->Destroy();
        TestFalse(TEXT("Lost roof stops episode instead of keeping airborne sensor"), Blue->CanAdvance(Error));
        Blue->Budget += 1;
        TestFalse(TEXT("Configuration mutation blocks advance"), Blue->CanAdvance(Error));
        TestTrue(TEXT("Reset clears prior committed layout"), Manager->BeginPlacementEpisode(124, Context, Error));
        TestFalse(TEXT("New episode needs new Blue commit"), Blue->IsCommitted());
        TestTrue(TEXT("Reset resolves underlying ground"), Blue->SiteWorldCm(0, Blue->Catalogue[0]).Equals(FVector(3000,0,400), .1));
        // Missing support must reject atomically even when a client ignores the mask.
        Ground->Destroy();
        TestTrue(TEXT("Reset without ground"), Manager->BeginPlacementEpisode(125, Context, Error));
        Action->SetStringField(TEXT("runId"), Context.RunId.ToString());
        Action->SetNumberField(TEXT("revision"), Context.Revision);
        TestFalse(TEXT("Airborne placement rejected"), Blue->DeployJson(*Action, Error)->GetBoolField(TEXT("accepted")));
        TestFalse(TEXT("Airborne placement never commits"), Blue->IsCommitted());
        // A ledge narrower than the mount and a steep roof are not legal surfaces.
        Roof = AddSurface(FVector(3000,0,950), FVector(.5,.5,1));
        Manager->BeginPlacementEpisode(126, Context, Error);
        TestTrue(TEXT("Narrow ledge blocked"), Blue->ContextJson()->GetArrayField(TEXT("siteSurfacesWorldCm"))[0]->IsNull());
        Roof->Destroy();
        Roof = AddSurface(FVector(3000,0,950), FVector(4,4,1));
        Roof->SetActorRotation(FRotator(30,0,0));
        Manager->BeginPlacementEpisode(127, Context, Error);
        TestTrue(TEXT("Steep roof blocked"), Blue->ContextJson()->GetArrayField(TEXT("siteSurfacesWorldCm"))[0]->IsNull());
    }
    GEngine->DestroyWorldContext(World);
    World->DestroyWorld(false);
    return true;
}
#endif
