#include "Simulation/BlueTeam/BlueTeamCoordinator.h"
#include "Simulation/RedTeam/RedTeamManager.h"
#include "Dom/JsonObject.h"
#include "Engine/Engine.h"
#include "Engine/TargetPoint.h"
#include "Engine/World.h"
#include "Engine/StaticMeshActor.h"
#include "Components/StaticMeshComponent.h"
#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS
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
        Action->SetNumberField(TEXT("schemaVersion"), 1);
        Action->SetStringField(TEXT("runId"), Context.RunId.ToString());
        Action->SetNumberField(TEXT("revision"), Context.Revision);
        Action->SetNumberField(TEXT("requestId"), 0);
        Action->SetNumberField(TEXT("expectedStep"), 0);
        Action->SetBoolField(TEXT("commit"), true);
        auto Placement = MakeShared<FJsonObject>();
        Placement->SetStringField(TEXT("profileId"), TEXT("unavailable-profile"));
        Placement->SetNumberField(TEXT("siteId"), 0);
        TArray<TSharedPtr<FJsonValue>> Rows;
        Rows.Add(MakeShared<FJsonValueObject>(Placement));
        Action->SetArrayField(TEXT("placements"), Rows);
        TestFalse(TEXT("Unknown profile rejected atomically"), Blue->DeployJson(*Action, Error)->GetBoolField(TEXT("accepted")));
        TestFalse(TEXT("Rejected action does not commit"), Blue->IsCommitted());
        Placement->SetStringField(TEXT("profileId"), TEXT("rf"));
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
