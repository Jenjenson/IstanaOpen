#include "Simulation/BlueTeam/BlueTeamCoordinator.h"
#include "Simulation/RedTeam/RedTeamManager.h"
#include "Components/StaticMeshComponent.h"
#include "Components/SceneComponent.h"
#include "Engine/StaticMesh.h"
#include "Engine/TargetPoint.h"
#include "Engine/World.h"
#include "UObject/ConstructorHelpers.h"
#include "DrawDebugHelpers.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "Policies/CondensedJsonPrintPolicy.h"

namespace
{
    using FValues = TArray<TSharedPtr<FJsonValue>>;
    TSharedPtr<FJsonValue> Number(double Value) { return MakeShared<FJsonValueNumber>(Value); }
    TSharedPtr<FJsonValue> String(const FString& Value) { return MakeShared<FJsonValueString>(Value); }
    FValues VectorArray(const FVector& Value) { return {Number(Value.X), Number(Value.Y), Number(Value.Z)}; }
    FValues VectorArray(const FVector2D& Value) { return {Number(Value.X), Number(Value.Y)}; }
    FString Encode(const TSharedRef<FJsonObject>& Object)
    {
        FString Text;
        FJsonSerializer::Serialize(Object, TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Text));
        return Text;
    }
    bool InRange(double V, double Lo, double Hi) { return FMath::IsFinite(V) && V >= Lo && V <= Hi; }
    bool Integer(const FJsonObject& Object, const TCHAR* Key, double Lo, double Hi, double& V)
    { return Object.TryGetNumberField(Key, V) && InRange(V, Lo, Hi) && FMath::FloorToDouble(V) == V; }
    TSharedRef<FJsonObject> ProfileJson(const FBlueSensorProfile& Profile)
    {
        auto Object = MakeShared<FJsonObject>();
        Object->SetStringField(TEXT("id"), Profile.Id); Object->SetStringField(TEXT("label"), Profile.Label);
        Object->SetNumberField(TEXT("cost"), Profile.Cost); Object->SetNumberField(TEXT("height_m"), Profile.HeightM);
        auto Ranges = MakeShared<FJsonObject>(); auto Strengths = MakeShared<FJsonObject>();
        const TCHAR* Modalities[] = {TEXT("rf"), TEXT("radar"), TEXT("eo"), TEXT("thermal")};
        for (int32 I = 0; I < 4; ++I)
        { Ranges->SetNumberField(Modalities[I], Profile.RangesM[I]); Strengths->SetNumberField(Modalities[I], Profile.Strengths[I]); }
        Object->SetObjectField(TEXT("ranges"), Ranges); Object->SetObjectField(TEXT("strengths"), Strengths);
        return Object;
    }
}

ABlueSensorMarker::ABlueSensorMarker()
{
    PrimaryActorTick.bCanEverTick = false;
    SetRootComponent(CreateDefaultSubobject<USceneComponent>(TEXT("SurfaceMount")));
    Body = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("SensorBody")); Body->SetupAttachment(RootComponent);
    static ConstructorHelpers::FObjectFinder<UStaticMesh> Cylinder(TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
    Body->SetStaticMesh(Cylinder.Object); Body->SetRelativeScale3D(FVector(.8, .8, 1.5));
    Body->SetCollisionEnabled(ECollisionEnabled::NoCollision); Body->SetCanEverAffectNavigation(false);
}

void ABlueSensorMarker::SetMastHeight(double HeightM)
{
    // The primitive is 100 cm tall, centred on its origin. Its bottom, not its
    // centre, rests on the sampled surface; the sensing point is at the top.
    const double HeightCm = FMath::Max(1., HeightM * 100.);
    Body->SetRelativeScale3D(FVector(.8, .8, HeightCm / 100.));
    Body->SetRelativeLocation(FVector(0, 0, HeightCm / 2.));
}

ABlueTeamCoordinator::ABlueTeamCoordinator()
{
    PrimaryActorTick.bCanEverTick = true;
    auto Add = [&](const TCHAR* Id, const TCHAR* Label, double Price, FVector4 Ranges, FVector4 Strengths)
    {
        FBlueSensorProfile Profile; Profile.Id = Id; Profile.Label = Label; Profile.Cost = Price;
        Profile.RangesM = Ranges; Profile.Strengths = Strengths;
        Catalogue.Add(Profile); AvailableSensorIds.Add(Id);
    };
    Add(TEXT("rf"), TEXT("Passive RF"), .8, FVector4(130,0,0,0), FVector4(.88,0,0,0));
    Add(TEXT("radar"), TEXT("Search radar"), 1.2, FVector4(0,100,0,0), FVector4(0,.86,0,0));
    Add(TEXT("eo"), TEXT("Electro-optical"), .7, FVector4(0,0,100,0), FVector4(0,0,.94,0));
    Add(TEXT("thermal"), TEXT("Thermal"), 1, FVector4(0,0,0,115), FVector4(0,0,0,.86));
    Add(TEXT("fused"), TEXT("Radar + thermal"), 2, FVector4(0,100,0,125), FVector4(0,.86,0,.86));
    for (double Radius : {30., 45.})
        for (int32 I = 0; I < 16; ++I)
        { const double Angle = I * PI / 8.; ApprovedSitesM.Add(FVector2D(FMath::Cos(Angle), FMath::Sin(Angle)) * Radius); }
    ApproachWeights.Init(.125, 8);
}

bool ABlueTeamCoordinator::ValidateConfiguration(double Step, FString& Error) const
{
    Error.Reset();
    auto Fail = [&](const TCHAR* Reason) { Error = Reason; return false; };
    if (!IsValid(Manager) || Manager->BlueCoordinator != this) return Fail(TEXT("Blue coordinator must reference its owning Red manager."));
    if (!InRange(Step, .001, .1) || !InRange(LookIntervalSeconds, .1, 3)
        || FMath::Abs(LookIntervalSeconds / Step - FMath::RoundToDouble(LookIntervalSeconds / Step)) > 1.e-7)
        return Fail(TEXT("Blue look interval must be an exact positive multiple of the fixed step."));
    if (!InRange(Budget, .001, 100) || MaxSites < 1 || MaxSites > 32
        || !InRange(MinimumSeparationM, 0, 1000) || !InRange(DeploymentMinRadiusM, 0, 1000)
        || !InRange(DeploymentMaxRadiusM, DeploymentMinRadiusM, 2000)
        || !InRange(ObjectiveRadiusM, 1, 100) || !InRange(DefenceLeadTimeSeconds, 0, 30)
        || !InRange(TimeLimitSeconds, .1, 300) || FMath::FloorToDouble(TimeLimitSeconds / LookIntervalSeconds) + 1 > 512
        || RequiredConfirmations < 1 || RequiredConfirmations > ConfirmationWindow || ConfirmationWindow > 8)
        return Fail(TEXT("Invalid Blue budget, site limits, timing or confirmation rules."));
    if (!InRange(PriorSpawnRadiusM, 180, 2000) || PriorSpawnRadiusM <= ObjectiveRadiusM
        || !InRange(PriorAltitudeM, 0, 1000) || !InRange(PriorSpeedMps, .1, 200)
        || !InRange(PriorEmitterProbability, 0, 1) || PriorSwarmSize < 1 || PriorSwarmSize > 64
        || ApproachWeights.Num() != 8) return Fail(TEXT("Invalid Blue public mission priors."));
    double WeightSum = 0;
    for (double Weight : ApproachWeights)
    { if (!InRange(Weight, 0, 1)) return Fail(TEXT("Invalid approach weights.")); WeightSum += Weight; }
    if (WeightSum <= 0) return Fail(TEXT("Approach weights require positive mass."));
    for (double Value : {Visibility, Rain, Illumination, Humidity, RFNoise})
        if (!InRange(Value, 0, 1)) return Fail(TEXT("Weather values must be in [0,1]."));
    if (Catalogue.Num() < 1 || Catalogue.Num() > 32 || ApprovedSitesM.Num() < 1 || ApprovedSitesM.Num() > 512)
        return Fail(TEXT("Blue requires 1..32 profiles and 1..512 approved sites."));
    TSet<FString> Ids;
    for (const auto& Profile : Catalogue)
    {
        if (Profile.Id.IsEmpty() || Profile.Id.Len() > 128 || Ids.Contains(Profile.Id)
            || !InRange(Profile.Cost, .001, 100) || !InRange(Profile.HeightM, 0, 200))
            return Fail(TEXT("Invalid or duplicate Blue profile ID, cost or height."));
        Ids.Add(Profile.Id); bool bRange = false;
        for (int32 I = 0; I < 4; ++I)
        {
            if (!InRange(Profile.RangesM[I], 0, 2000) || !InRange(Profile.Strengths[I], 0, 1))
                return Fail(TEXT("Invalid Blue profile capability."));
            bRange |= Profile.RangesM[I] > 0;
        }
        if (!bRange) return Fail(TEXT("Blue profiles require a positive sensing range."));
    }
    for (const auto& Id : AvailableSensorIds) if (!Ids.Contains(Id)) return Fail(TEXT("Unknown available sensor ID."));
    for (const auto& Site : ApprovedSitesM)
        if (!InRange(Site.X, -10000, 10000) || !InRange(Site.Y, -10000, 10000)) return Fail(TEXT("Invalid approved site."));
    for (int32 Site : BlockedSites) if (!ApprovedSitesM.IsValidIndex(Site)) return Fail(TEXT("Unknown blocked site."));
    return true;
}

FString ABlueTeamCoordinator::ConfigurationSignature() const
{
    // Reuse the public serialization, removing only episode state. Changes require reset.
    auto Object = ContextJson();
    Object->RemoveField(TEXT("completedSteps")); Object->RemoveField(TEXT("committed"));
    const auto Snapshot = Object->GetObjectField(TEXT("publicSnapshot"));
    for (const TCHAR* Key : {TEXT("timestamp"), TEXT("placements"), TEXT("budget_remaining"), TEXT("tracks"), TEXT("done"), TEXT("fresh_track_fraction")})
        Snapshot->RemoveField(Key);
    return Encode(Object);
}

void ABlueTeamCoordinator::BeginEpisode(const FRedTeamPlacementContext& Context)
{
    ClearMarkers(); Placements.Reset(); Evidence.Reset(); bCommitted = false; Cost = 0;
    CompletedSteps = 0; LastActionId = -1; LastActionPayload.Reset(); LastActionResult.Reset(); NextTrackId = 0;
    RunId = Context.RunId; Revision = Context.Revision; OriginWorldCm = Context.ObjectiveWorldCm;
    EpisodeSeed = Context.MemberSeed; FixedStepSeconds = Context.FixedStepSeconds;
    LookSteps = FMath::RoundToInt(LookIntervalSeconds / FixedStepSeconds);
    SiteSurfacesCm.SetNumZeroed(ApprovedSitesM.Num());
    SupportedSites.SetNumZeroed(ApprovedSitesM.Num());
    for (int32 I = 0; I < ApprovedSitesM.Num(); ++I)
        SupportedSites[I] = ResolveSurfaceCm(I, SiteSurfacesCm[I]);
    FrozenConfiguration = ConfigurationSignature();
}

void ABlueTeamCoordinator::CaptureInitialStates(const TArray<FIstanaDroneState>& States)
{
    Evidence.Reset();
    for (const auto& State : States)
    {
        auto& Item = Evidence.Add(State.DroneId);
        Item.PreviousPosition = State.PositionCm; Item.bHasPrevious = true;
        if (FVector::Dist2D(State.PositionCm, OriginWorldCm) <= ObjectiveRadiusM * 100) Item.ZoneEntry = 0;
    }
}

bool ABlueTeamCoordinator::CanAdvance(FString& Error) const
{
    Error.Reset();
    if (!IsValid(Manager) || Manager->BlueCoordinator != this || !RunId.IsValid()
        || Manager->GetPlacementContext().RunId != RunId || Manager->GetPlacementContext().Revision != Revision)
        Error = TEXT("Blue context unavailable or stale; reset the episode.");
    else if (FrozenConfiguration != ConfigurationSignature()
        || !IsValid(Manager->ObjectiveTarget) || Manager->ObjectiveTarget->GetActorLocation() != OriginWorldCm
        || Manager->PlacementWorldRevision != Manager->GetPlacementContext().WorldRevision)
        Error = TEXT("Blue configuration, objective or world changed; reset the episode.");
    else if (!bCommitted) Error = TEXT("Blue initial sensor layout must be committed before stepping.");
    else for (const auto& Placement : Placements)
        if (!SurfaceUnchanged(Placement.SiteId))
        { Error = TEXT("Blue supporting surface changed or disappeared; reset the episode."); break; }
    return Error.IsEmpty();
}

bool ABlueTeamCoordinator::ResolveSurfaceCm(int32 SiteId, FVector& Surface) const
{
    if (!GetWorld() || !ApprovedSitesM.IsValidIndex(SiteId)) return false;
    const auto Site = ApprovedSitesM[SiteId];
    const FVector Centre = OriginWorldCm + FVector(Site.X * 100, Site.Y * 100, 0);
    FCollisionQueryParams Query(SCENE_QUERY_STAT(BlueSurfaceSupport), true, this);
    Query.AddIgnoredActor(Manager);
    FCollisionObjectQueryParams Objects(ECC_WorldStatic);
    // Centre + rim of an 80 cm mount. Reject edges, steep faces and missing
    // collision; never invent a flat ground plane when geometry is absent.
    double Lowest = DBL_MAX, Highest = -DBL_MAX;
    for (const FVector2D Offset : {FVector2D(0,0), FVector2D(40,0), FVector2D(-40,0), FVector2D(0,40), FVector2D(0,-40)})
    {
        const FVector XY = Centre + FVector(Offset.X, Offset.Y, 0);
        FHitResult Hit;
        if (!GetWorld()->LineTraceSingleByObjectType(Hit, XY + FVector(0,0,100000), XY - FVector(0,0,100000), Objects, Query)
            || Hit.bStartPenetrating || Hit.ImpactNormal.Z < .985
            || !Hit.GetComponent() || Hit.GetComponent()->Mobility != EComponentMobility::Static)
            return false;
        Lowest = FMath::Min(Lowest, Hit.ImpactPoint.Z);
        Highest = FMath::Max(Highest, Hit.ImpactPoint.Z);
    }
    if (Highest - Lowest > 10.) return false;
    // Slight embedding on gently uneven surfaces is preferable to a floating base.
    Surface = FVector(Centre.X, Centre.Y, Lowest);
    return true;
}

bool ABlueTeamCoordinator::SurfaceUnchanged(int32 SiteId) const
{
    FVector Current;
    return SupportedSites.IsValidIndex(SiteId) && SupportedSites[SiteId]
        && ResolveSurfaceCm(SiteId, Current) && Current.Equals(SiteSurfacesCm[SiteId], .1);
}

FVector ABlueTeamCoordinator::SiteWorldCm(int32 SiteId, const FBlueSensorProfile& Profile) const
{
    check(SupportedSites.IsValidIndex(SiteId) && SupportedSites[SiteId]);
    return SiteSurfacesCm[SiteId] + FVector(0, 0, Profile.HeightM * 100);
}

TSharedRef<FJsonObject> ABlueTeamCoordinator::DeployJson(const FJsonObject& Action, FString& Error)
{
    auto Result = MakeShared<FJsonObject>(); Result->SetBoolField(TEXT("accepted"), false);
    Result->SetBoolField(TEXT("committed"), bCommitted); Result->SetNumberField(TEXT("cost"), Cost);
    auto Fail = [&](const TCHAR* Why) { Error = Why; Result->SetStringField(TEXT("error"), Error); return Result; };
    Error.Reset(); double Schema, ActionRevision, RequestId, ExpectedStep;
    FString ActionRun; FGuid ParsedRun; bool bCommit = false;
    const TArray<TSharedPtr<FJsonValue>>* Entries = nullptr;
    if (!Integer(Action, TEXT("schemaVersion"), 1, 1, Schema)
        || !Integer(Action, TEXT("revision"), 0, 9007199254740991., ActionRevision)
        || !Integer(Action, TEXT("requestId"), 0, 9007199254740991., RequestId)
        || !Integer(Action, TEXT("expectedStep"), 0, 0, ExpectedStep)
        || !Action.TryGetStringField(TEXT("runId"), ActionRun) || !FGuid::Parse(ActionRun, ParsedRun)
        || !Action.TryGetBoolField(TEXT("commit"), bCommit) || !bCommit
        || !Action.TryGetArrayField(TEXT("placements"), Entries) || Entries->Num() > 32)
        return Fail(TEXT("Invalid Blue action: schemaVersion, runId, revision, requestId, expectedStep=0, placements and commit=true required."));
    if (ParsedRun != RunId || int64(ActionRevision) != Revision || !RunId.IsValid()) return Fail(TEXT("Stale Blue runId/revision."));
    // Canonicalize only supported semantic fields so object-key ordering is irrelevant.
    TArray<FBlueSensorPlacement> Proposed;
    FString Payload = FString::Printf(TEXT("%lld/%lld/"), int64(ActionRevision), int64(RequestId));
    for (const auto& Value : *Entries)
    {
        const TSharedPtr<FJsonObject>* Entry = nullptr; FString Id; double Site;
        if (!Value->TryGetObject(Entry) || !(*Entry)->TryGetStringField(TEXT("profileId"), Id)
            || !Integer(**Entry, TEXT("siteId"), 0, 511, Site)) return Fail(TEXT("Each Blue placement needs profileId and integer siteId."));
        FBlueSensorPlacement Placement; Placement.ProfileId = Id; Placement.SiteId = int32(Site); Proposed.Add(Placement);
        Payload += FString::Printf(TEXT("%d:%s:%d/"), Id.Len(), *Id, Placement.SiteId);
    }
    if (int64(RequestId) == LastActionId)
    {
        if (Payload != LastActionPayload) return Fail(TEXT("Blue requestId reused with different payload."));
        if (LastActionResult.IsValid()) return LastActionResult.ToSharedRef();
    }
    if (int64(RequestId) <= LastActionId) return Fail(TEXT("Stale Blue requestId."));
    if (!IsValid(Manager) || Manager->GetPlacementContext().RunId != RunId
        || (Manager->EpisodePhase != ERedTeamEpisodePhase::AwaitingPlacement && Manager->EpisodePhase != ERedTeamEpisodePhase::Running)
        || CompletedSteps != 0 || Manager->GetEpisodeObservation().CompletedSteps != 0 || bCommitted)
        return Fail(TEXT("Blue accepts one initial layout before clock advance."));
    if (FrozenConfiguration != ConfigurationSignature() || !IsValid(Manager->ObjectiveTarget)
        || Manager->ObjectiveTarget->GetActorLocation() != OriginWorldCm
        || Manager->PlacementWorldRevision != Manager->GetPlacementContext().WorldRevision)
        return Fail(TEXT("Blue context changed; reset the episode."));
    if (Proposed.Num() > MaxSites) return Fail(TEXT("Blue layout exceeds maximum sites."));
    TSet<int32> Seen; double ProposedCost = 0;
    for (const auto& Placement : Proposed)
    {
        const auto* Profile = Catalogue.FindByPredicate([&](const auto& P) { return P.Id == Placement.ProfileId; });
        if (!Profile || !AvailableSensorIds.Contains(Placement.ProfileId)) return Fail(TEXT("Unknown or unavailable Blue profile."));
        if (!ApprovedSitesM.IsValidIndex(Placement.SiteId) || BlockedSites.Contains(Placement.SiteId) || Seen.Contains(Placement.SiteId))
            return Fail(TEXT("Blue site is unknown, blocked or duplicated."));
        if (!SurfaceUnchanged(Placement.SiteId))
            return Fail(TEXT("Blue site has no stable supporting surface; reset and choose a supported site."));
        const auto Site = ApprovedSitesM[Placement.SiteId]; const double Radius = Site.Size();
        if (Radius < DeploymentMinRadiusM - 1.e-8 || Radius > DeploymentMaxRadiusM + 1.e-8)
            return Fail(TEXT("Blue site outside approved deployment radii."));
        for (int32 Other : Seen)
            if (FVector2D::Distance(Site, ApprovedSitesM[Other]) < MinimumSeparationM - 1.e-8)
                return Fail(TEXT("Blue sites violate minimum separation."));
        Seen.Add(Placement.SiteId); ProposedCost += Profile->Cost;
    }
    if (ProposedCost > Budget + 1.e-8) return Fail(TEXT("Blue layout exceeds budget."));
    Placements = MoveTemp(Proposed); Cost = ProposedCost; bCommitted = true;
    if (bSpawnSensorMarkers && GetWorld())
        for (const auto& Placement : Placements)
        {
            const auto* Profile = Catalogue.FindByPredicate([&](const auto& P) { return P.Id == Placement.ProfileId; });
            auto* Marker = GetWorld()->SpawnActor<ABlueSensorMarker>(SiteSurfacesCm[Placement.SiteId], FRotator::ZeroRotator);
            if (Marker) { Marker->SetMastHeight(Profile->HeightM); Markers.Add(Marker); }
        }
    Result->SetBoolField(TEXT("accepted"), true); Result->SetBoolField(TEXT("committed"), true);
    Result->SetNumberField(TEXT("cost"), Cost);
    LastActionId = int64(RequestId); LastActionPayload = Payload; LastActionResult = Result;
    return Result;
}

double ABlueTeamCoordinator::Uniform(int32 DroneId, int32 SensorId, int64 Look, uint32 Salt) const
{
    // Integer-only indexed RNG: unaffected by draw order, batching, run GUIDs or rendering.
    uint32 Value = uint32(EpisodeSeed) ^ Salt;
    for (uint32 Part : {uint32(DroneId), uint32(SensorId), uint32(Look), uint32(uint64(Look) >> 32)})
    { Value ^= Part + 0x9e3779b9u + (Value << 6) + (Value >> 2); Value ^= Value >> 16; Value *= 0x7feb352du; Value ^= Value >> 15; }
    return (double(Value) + .5) / 4294967296.;
}

double ABlueTeamCoordinator::DetectionProbability(const FBlueSensorProfile& Profile, const FVector& Sensor,
    const FVector& Target, bool bEmitting) const
{
    const double Distance = FVector::Distance(Sensor, Target) / 100.;
    // Deliberately analytical: no terrain, occlusion, false positives or classified sensor data.
    const double Weather[] = {bEmitting ? 1 - .7 * RFNoise : 0, 1 - .4 * Rain,
        Visibility * (.15 + .85 * Illumination) * (1 - .5 * Rain), (1 - .45 * Humidity) * (1 - .3 * Rain)};
    double Miss = 1;
    for (int32 I = 0; I < 4; ++I)
    {
        const double Range = Profile.RangesM[I];
        if (Range <= 0 || Distance > Range) continue;
        const double Probability = FMath::Clamp(Profile.Strengths[I] * Weather[I] * (1 - .5 * FMath::Square(Distance / Range)), 0., 1.);
        Miss *= 1 - Probability;
    }
    return 1 - Miss;
}

void ABlueTeamCoordinator::Evaluate(FRedTeamEpisodeObservation& Observation)
{
    if (Observation.RunId != RunId || Observation.CompletedSteps <= CompletedSteps) return;
    CompletedSteps = Observation.CompletedSteps;
    const double Now = CompletedSteps * FixedStepSeconds;
    const int64 Look = CompletedSteps / LookSteps;
    int32 Entered = 0;
    for (const auto& Drone : Observation.Drones)
    {
        auto& Item = Evidence.FindOrAdd(Drone.DroneId);
        const FVector Relative = Drone.PositionCm - OriginWorldCm;
        const double RadiusCm = ObjectiveRadiusM * 100;
        if (Item.ZoneEntry < 0 && Item.bHasPrevious)
        {
            // Segment-circle crossing prevents skipping the zone between movement steps.
            const FVector2D A(Item.PreviousPosition.X - OriginWorldCm.X, Item.PreviousPosition.Y - OriginWorldCm.Y);
            const FVector2D D(Drone.PositionCm.X - Item.PreviousPosition.X, Drone.PositionCm.Y - Item.PreviousPosition.Y);
            const double C = A.SizeSquared() - RadiusCm * RadiusCm;
            if (C <= 0) Item.ZoneEntry = Now - FixedStepSeconds;
            else if (D.SizeSquared() > SMALL_NUMBER)
            {
                const double B = 2 * FVector2D::DotProduct(A, D); const double Disc = B * B - 4 * D.SizeSquared() * C;
                if (Disc >= 0)
                {
                    const double T = (-B - FMath::Sqrt(Disc)) / (2 * D.SizeSquared());
                    if (T >= 0 && T <= 1) Item.ZoneEntry = Now - FixedStepSeconds + T * FixedStepSeconds;
                }
            }
        }
        else if (Item.ZoneEntry < 0 && Relative.Size2D() <= RadiusCm) Item.ZoneEntry = Now;
        Item.PreviousPosition = Drone.PositionCm; Item.bHasPrevious = true;
        // One fused hit per target per sensing look, independent of sensor count.
        if (Drone.bActive && CompletedSteps % LookSteps == 0 && (Item.ZoneEntry < 0 || Now <= Item.ZoneEntry + 1.e-8))
        {
            bool bHit = false;
            const bool bEmitting = Uniform(Drone.DroneId, -1, 0, 0x18231u) < .5;
            for (const auto& Placement : Placements)
            {
                const auto* Profile = Catalogue.FindByPredicate([&](const auto& P) { return P.Id == Placement.ProfileId; });
                if (Uniform(Drone.DroneId, Placement.SiteId, Look, 0x763afu) < DetectionProbability(*Profile, SiteWorldCm(Placement.SiteId, *Profile), Drone.PositionCm, bEmitting)) bHit = true;
            }
            Item.HitLooks.RemoveAll([&](int64 Previous) { return Previous <= Look - ConfirmationWindow; });
            if (bHit)
            {
                Item.HitLooks.Add(Look);
                if (Item.FirstDetection < 0) { Item.FirstDetection = Now; Item.PublicTrackId = NextTrackId++; }
                if (Item.FirstConfirmation < 0 && Item.HitLooks.Num() >= RequiredConfirmations) Item.FirstConfirmation = Now;
                Item.ReportTime = Now;
                // Quantized reported measurements are updated only on successful looks.
                const FVector Position = Relative / 100., Velocity = Drone.VelocityCmPerSecond / 100.;
                for (int32 I = 0; I < 3; ++I)
                { Item.ReportPositionM[I] = FMath::RoundToDouble(Position[I]); Item.ReportVelocityMps[I] = FMath::RoundToDouble(Velocity[I] * 2) / 2; }
            }
        }
        if (Item.ZoneEntry >= 0) ++Entered;
    }
    if (Observation.Drones.Num() > 0 && Entered == Observation.Drones.Num())
    { Observation.bTerminated = true; Observation.Reason = TEXT("All threats entered the objective zone; abstract sensing outcomes evaluated."); }
    else if (Now + 1.e-8 >= TimeLimitSeconds)
    { Observation.bTruncated = true; Observation.Reason = TEXT("Blue episode time limit; unentered threats remain unresolved."); }
    if (Observation.bTerminated || Observation.bTruncated)
    {
        // Red reward is the negative of terminal Blue sensing return; neither represents interception.
        double Detected = 0, Confirmed = 0, Timely = 0, Breached = 0;
        for (const auto& Pair : Evidence)
        {
            const auto& Item = Pair.Value;
            Detected += Item.FirstDetection >= 0; Confirmed += Item.FirstConfirmation >= 0;
            const bool bTimely = Item.ZoneEntry >= 0 && Item.FirstConfirmation >= 0
                && Item.FirstConfirmation <= Item.ZoneEntry - DefenceLeadTimeSeconds + 1.e-8;
            Timely += bTimely; Breached += Item.ZoneEntry >= 0 && !bTimely;
        }
        const double Count = FMath::Max(1, Evidence.Num());
        Observation.bHasReward = true; Observation.Reward = -(2 * Detected / Count + 3 * Confirmed / Count
            + 5 * Timely / Count - 5 * Breached / Count - Cost / Budget);
    }
}

TSharedRef<FJsonObject> ABlueTeamCoordinator::PublicSnapshotJson() const
{
    auto Object = MakeShared<FJsonObject>();
    const double Now = CompletedSteps * FixedStepSeconds;
    Object->SetStringField(TEXT("schema"), TEXT("triad.sensor_input.v1"));
    Object->SetStringField(TEXT("source"), TEXT("istana-synthetic-blue-v1"));
    Object->SetStringField(TEXT("episode_id"), RunId.ToString()); Object->SetNumberField(TEXT("timestamp"), Now);
    Object->SetNumberField(TEXT("max_track_age"), 10);
    FValues Sites, Placed, Available, Blocked, Tracks, Weights;
    for (const auto& Site : ApprovedSitesM) Sites.Add(MakeShared<FJsonValueArray>(VectorArray(Site)));
    for (const auto& Placement : Placements)
    {
        const int32 Index = Catalogue.IndexOfByPredicate([&](const auto& P) { return P.Id == Placement.ProfileId; });
        if (Index == INDEX_NONE || !ApprovedSitesM.IsValidIndex(Placement.SiteId)) continue;
        auto Row = MakeShared<FJsonObject>(); Row->SetStringField(TEXT("sensor_id"), Placement.ProfileId);
        Row->SetNumberField(TEXT("sensor_index"), Index); Row->SetNumberField(TEXT("cost"), Catalogue[Index].Cost);
        Row->SetArrayField(TEXT("position"), VectorArray(ApprovedSitesM[Placement.SiteId])); Placed.Add(MakeShared<FJsonValueObject>(Row));
    }
    for (const auto& Id : AvailableSensorIds) Available.Add(String(Id));
    for (int32 Index = 0; Index < ApprovedSitesM.Num(); ++Index)
        if (BlockedSites.Contains(Index) || !SupportedSites.IsValidIndex(Index) || !SupportedSites[Index]) Blocked.Add(Number(Index));
    TArray<const FTargetEvidence*> Reports;
    for (const auto& Pair : Evidence) if (Pair.Value.ReportTime >= 0 && Now - Pair.Value.ReportTime <= 10) Reports.Add(&Pair.Value);
    Reports.Sort([](const FTargetEvidence& A, const FTargetEvidence& B) { return A.PublicTrackId < B.PublicTrackId; });
    for (const auto* Item : Reports)
    {
        auto Track = MakeShared<FJsonObject>(); Track->SetStringField(TEXT("id"), FString::Printf(TEXT("report-%d"), Item->PublicTrackId));
        Track->SetArrayField(TEXT("position"), VectorArray(Item->ReportPositionM));
        Track->SetArrayField(TEXT("velocity"), VectorArray(Item->ReportVelocityMps));
        Track->SetNumberField(TEXT("confidence"), Item->FirstConfirmation >= 0 ? .9 : .6);
        Track->SetNumberField(TEXT("emitter_probability"), PriorEmitterProbability);
        Track->SetNumberField(TEXT("timestamp"), Item->ReportTime); Track->SetBoolField(TEXT("confirmed"), Item->FirstConfirmation >= 0);
        Tracks.Add(MakeShared<FJsonValueObject>(Track));
    }
    for (double Weight : ApproachWeights) Weights.Add(Number(Weight));
    Object->SetArrayField(TEXT("sites"), Sites); Object->SetArrayField(TEXT("placements"), Placed);
    Object->SetArrayField(TEXT("available_sensor_ids"), Available); Object->SetArrayField(TEXT("blocked_sites"), Blocked);
    Object->SetArrayField(TEXT("tracks"), Tracks); Object->SetNumberField(TEXT("fresh_track_fraction"), Reports.Num() ? 1 : 0);
    Object->SetNumberField(TEXT("budget_total"), Budget); Object->SetNumberField(TEXT("budget_remaining"), FMath::Max(0., Budget - Cost));
    Object->SetNumberField(TEXT("max_sites"), MaxSites); Object->SetNumberField(TEXT("min_separation"), MinimumSeparationM);
    Object->SetNumberField(TEXT("deployment_min_radius"), DeploymentMinRadiusM); Object->SetNumberField(TEXT("deployment_max_radius"), DeploymentMaxRadiusM);
    auto Weather = MakeShared<FJsonObject>(); Weather->SetNumberField(TEXT("visibility"), Visibility);
    Weather->SetNumberField(TEXT("rain"), Rain); Weather->SetNumberField(TEXT("illumination"), Illumination);
    Weather->SetNumberField(TEXT("humidity"), Humidity); Weather->SetNumberField(TEXT("rf_noise"), RFNoise);
    Object->SetObjectField(TEXT("weather"), Weather);
    auto Forecast = MakeShared<FJsonObject>(); Forecast->SetArrayField(TEXT("approach_weights"), Weights);
    Forecast->SetNumberField(TEXT("altitude"), PriorAltitudeM); Forecast->SetNumberField(TEXT("speed"), PriorSpeedMps);
    Forecast->SetNumberField(TEXT("emitter_probability"), PriorEmitterProbability); Forecast->SetNumberField(TEXT("swarm_size"), PriorSwarmSize);
    Forecast->SetNumberField(TEXT("angular_uncertainty"), .4); Object->SetObjectField(TEXT("forecast"), Forecast);
    Object->SetBoolField(TEXT("done"), IsValid(Manager) && (Manager->EpisodePhase == ERedTeamEpisodePhase::Completed || Manager->EpisodePhase == ERedTeamEpisodePhase::Cancelled));
    return Object;
}

TSharedRef<FJsonObject> ABlueTeamCoordinator::ContextJson() const
{
    auto Object = MakeShared<FJsonObject>(); Object->SetStringField(TEXT("runId"), RunId.ToString());
    Object->SetNumberField(TEXT("revision"), Revision); Object->SetNumberField(TEXT("completedSteps"), CompletedSteps);
    Object->SetBoolField(TEXT("committed"), bCommitted); Object->SetStringField(TEXT("coordinateSystem"), TEXT("unreal_xy_relative_m_z_up"));
    auto Origin = MakeShared<FJsonObject>(); Origin->SetNumberField(TEXT("x"), OriginWorldCm.X);
    Origin->SetNumberField(TEXT("y"), OriginWorldCm.Y); Origin->SetNumberField(TEXT("z"), OriginWorldCm.Z);
    Object->SetObjectField(TEXT("worldOriginCm"), Origin); Object->SetObjectField(TEXT("publicSnapshot"), PublicSnapshotJson());
    FValues Profiles; for (const auto& Profile : Catalogue) Profiles.Add(MakeShared<FJsonValueObject>(ProfileJson(Profile)));
    Object->SetArrayField(TEXT("catalogue"), Profiles);
    FValues Surfaces;
    for (int32 I = 0; I < ApprovedSitesM.Num(); ++I)
        Surfaces.Add(SupportedSites.IsValidIndex(I) && SupportedSites[I]
            ? TSharedPtr<FJsonValue>(MakeShared<FJsonValueArray>(VectorArray(SiteSurfacesCm[I])))
            : TSharedPtr<FJsonValue>(MakeShared<FJsonValueNull>()));
    Object->SetArrayField(TEXT("siteSurfacesWorldCm"), Surfaces);
    Object->SetStringField(TEXT("placementRule"), TEXT("static_surface_mast_v1"));
    auto Temporal = MakeShared<FJsonObject>(); Temporal->SetNumberField(TEXT("objective_radius_m"), ObjectiveRadiusM);
    Temporal->SetNumberField(TEXT("lead_time_s"), DefenceLeadTimeSeconds); Temporal->SetNumberField(TEXT("look_interval_s"), LookIntervalSeconds);
    Temporal->SetNumberField(TEXT("required_confirmations"), RequiredConfirmations); Temporal->SetNumberField(TEXT("confirmation_window"), ConfirmationWindow);
    Temporal->SetNumberField(TEXT("horizon_s"), TimeLimitSeconds); Temporal->SetNumberField(TEXT("prior_spawn_radius_m"), PriorSpawnRadiusM);
    Temporal->SetNumberField(TEXT("altitude_uncertainty_m"), 10); Temporal->SetNumberField(TEXT("max_hypotheses"), 64);
    Temporal->SetNumberField(TEXT("rf_persistent_weight"), .5); Object->SetObjectField(TEXT("temporalConfig"), Temporal);
    Object->SetNumberField(TEXT("fixedStepSeconds"), FixedStepSeconds); Object->SetNumberField(TEXT("timeLimitSeconds"), TimeLimitSeconds);
    Object->SetStringField(TEXT("sensorModel"), TEXT("surface-mounted mast; synthetic radial falloff and weather; no sensing occlusion"));
    return Object;
}

TSharedRef<FJsonObject> ABlueTeamCoordinator::ObservationJson() const
{
    auto Object = MakeShared<FJsonObject>(); Object->SetStringField(TEXT("runId"), RunId.ToString());
    Object->SetNumberField(TEXT("completedSteps"), CompletedSteps); Object->SetNumberField(TEXT("elapsedSeconds"), CompletedSteps * FixedStepSeconds);
    Object->SetBoolField(TEXT("committed"), bCommitted);
    const auto Red = IsValid(Manager) ? Manager->GetEpisodeObservation() : FRedTeamEpisodeObservation();
    const bool bTerminated = Red.RunId == RunId && Red.bTerminated, bTruncated = Red.RunId == RunId && Red.bTruncated;
    Object->SetBoolField(TEXT("terminated"), bTerminated); Object->SetBoolField(TEXT("truncated"), bTruncated);
    Object->SetStringField(TEXT("reason"), Red.RunId == RunId ? Red.Reason : TEXT("Awaiting reset"));
    const bool bMetrics = (bTerminated || bTruncated) && !Evidence.IsEmpty();
    Object->SetBoolField(TEXT("metricsAvailable"), bMetrics); auto Metrics = MakeShared<FJsonObject>(); double Reward = 0;
    if (bMetrics)
    {
        double Detected = 0, Confirmed = 0, Timely = 0, Breached = 0, Entered = 0;
        for (const auto& Pair : Evidence)
        {
            const auto& Item = Pair.Value; Detected += Item.FirstDetection >= 0; Confirmed += Item.FirstConfirmation >= 0;
            const bool bTimely = Item.ZoneEntry >= 0 && Item.FirstConfirmation >= 0 && Item.FirstConfirmation <= Item.ZoneEntry - DefenceLeadTimeSeconds + 1.e-8;
            Timely += bTimely; Breached += Item.ZoneEntry >= 0 && !bTimely; Entered += Item.ZoneEntry >= 0;
        }
        const double Count = Evidence.Num();
        Metrics->SetNumberField(TEXT("detected_fraction"), Detected / Count); Metrics->SetNumberField(TEXT("confirmed_fraction"), Confirmed / Count);
        Metrics->SetNumberField(TEXT("timely_fraction"), Timely / Count); Metrics->SetNumberField(TEXT("breached_fraction"), Breached / Count);
        Metrics->SetNumberField(TEXT("unresolved_fraction"), (Count - Entered) / Count); Metrics->SetNumberField(TEXT("cost"), Cost);
        Reward = 2 * Detected / Count + 3 * Confirmed / Count + 5 * Timely / Count - 5 * Breached / Count - Cost / Budget;
    }
    Object->SetObjectField(TEXT("metrics"), Metrics); Object->SetNumberField(TEXT("reward"), Reward);
    Object->SetObjectField(TEXT("publicSnapshot"), PublicSnapshotJson()); return Object;
}

void ABlueTeamCoordinator::ClearMarkers()
{ for (const auto& Marker : Markers) if (IsValid(Marker)) Marker->Destroy(); Markers.Reset(); }
void ABlueTeamCoordinator::Destroyed() { ClearMarkers(); Super::Destroyed(); }
void ABlueTeamCoordinator::EndPlay(const EEndPlayReason::Type Reason) { ClearMarkers(); Super::EndPlay(Reason); }
void ABlueTeamCoordinator::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);
    if (!bDrawCoverage || !GetWorld() || !IsValid(Manager) || Manager->GetPlacementContext().RunId != RunId) return;
    for (const auto& Placement : Placements)
    {
        const auto* Profile = Catalogue.FindByPredicate([&](const auto& P) { return P.Id == Placement.ProfileId; });
        if (!Profile || !ApprovedSitesM.IsValidIndex(Placement.SiteId)) continue;
        const FVector Position = SiteWorldCm(Placement.SiteId, *Profile);
        DrawDebugString(GetWorld(), Position + FVector(0,0,130), Profile->Label, nullptr, FColor::Cyan, 0, true);
        double Range = 0; for (int32 I = 0; I < 4; ++I) Range = FMath::Max(Range, double(Profile->RangesM[I]));
        DrawDebugCircle(GetWorld(), Position, Range * 100, 64, FColor(30,130,255), false, -1, 0, 4, FVector(1,0,0), FVector(0,1,0), false);
    }
    DrawDebugCircle(GetWorld(), OriginWorldCm, ObjectiveRadiusM * 100, 64, FColor::Orange, false, -1, 0, 8, FVector(1,0,0), FVector(0,1,0), false);
}
