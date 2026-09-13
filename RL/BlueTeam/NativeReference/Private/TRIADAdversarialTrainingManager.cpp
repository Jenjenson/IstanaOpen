#include "TRIADAdversarialTrainingManager.h"

#include "CesiumGeoreference.h"
#include "Dom/JsonObject.h"
#include "DrawDebugHelpers.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "HAL/FileManager.h"
#include "JsonObjectConverter.h"
#include "Misc/CommandLine.h"
#include "Misc/FileHelper.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "Misc/SecureHash.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "TRIADDemoDroneActor.h"
#include "TRIADLongRangeSensorModel.h"
#include "TRIADProtectedZoneComponent.h"
#include "TRIADRFEmitterComponent.h"
#include "TRIADRLTrainingModel.h"
#include "TRIADSensorNodeActor.h"
#include "TRIADSensorNodeComponent.h"
#include "TRIADSwarmControllerComponent.h"

namespace
{
constexpr int32 MaximumRLTargetActors = 64;
}

ATRIADAdversarialTrainingManager::ATRIADAdversarialTrainingManager()
{
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.bStartWithTickEnabled = false;
    ProtectedZone = CreateDefaultSubobject<UTRIADProtectedZoneComponent>(TEXT("ProtectedZone"));
    Tags.AddUnique(TEXT("TRIADSimulationOnlyRL"));
}

bool ATRIADAdversarialTrainingManager::IsTrainingRequested()
{
    return FParse::Param(FCommandLine::Get(), TEXT("TRIADRLTraining")) || IsEvaluationRequested();
}

bool ATRIADAdversarialTrainingManager::IsEvaluationRequested()
{
    return FParse::Param(FCommandLine::Get(), TEXT("TRIADRLEvaluation"));
}

bool ATRIADAdversarialTrainingManager::LoadTrainingConfig(
    FTRIADRLTrainingConfig& OutConfig, FString& OutError, FString* OutConfigFingerprint)
{
    // The explicit file is authoritative. A stale Content Browser asset must not
    // silently shadow a training/evaluation experiment's requested configuration.
    {
        FString ConfigPath = FPaths::Combine(
            FPaths::ProjectContentDir(), TEXT("TRIAD"), TEXT("RL"), TEXT("DefaultTrainingConfig.json"));
        FString OverridePath;
        if (FParse::Value(FCommandLine::Get(), TEXT("-TRIADRLConfig="), OverridePath))
        {
            ConfigPath = FPaths::ConvertRelativePathToFull(OverridePath);
        }
        FString JsonText;
        if (!FFileHelper::LoadFileToString(JsonText, *ConfigPath))
        {
            OutError = FString::Printf(TEXT("Could not load requested RL JSON configuration '%s'."), *ConfigPath);
            return false;
        }
        if (OutConfigFingerprint)
        {
            *OutConfigFingerprint = TEXT("md5:") + LexToString(FMD5Hash::HashFile(*ConfigPath)).ToLower();
        }
        FText FailureReason;
        if (!FJsonObjectConverter::JsonObjectStringToUStruct(JsonText, &OutConfig, 0, 0, false, &FailureReason))
        {
            OutError = TEXT("Could not deserialize RL definition: ") + FailureReason.ToString();
            return false;
        }
    }
    if (!OutConfig.bEnabled)
    {
        OutError = TEXT("RL definition is disabled.");
        return false;
    }
    return FTRIADRLTrainingModel::ValidateConfig(OutConfig, OutError);
}

void ATRIADAdversarialTrainingManager::BeginPlay()
{
    Super::BeginPlay();
    if (!IsTrainingRequested())
    {
        Destroy();
        return;
    }

    FString Error;
    if (!LoadTrainingConfig(TrainingConfig, Error, &ConfigFingerprint))
    {
        UE_LOG(LogTemp, Error, TEXT("TRIAD RL training did not start: %s"), *Error);
        Destroy();
        return;
    }
    Georeference = FindGeoreference();
    if (!Georeference)
    {
        UE_LOG(LogTemp, Error, TEXT("TRIAD RL training requires an existing CesiumGeoreference."));
        Destroy();
        return;
    }
    ProtectedZone->Configure(TrainingConfig.ProtectedZone);
    bEvaluationOnly = IsEvaluationRequested();
    SetActorTickEnabled(bEvaluationOnly || FParse::Param(FCommandLine::Get(), TEXT("TRIADRLDebug")));
    FTRIADRLStepResult InitialResult;
    if (!ResetEpisode(TrainingConfig.RandomSeed, InitialResult, Error))
    {
        UE_LOG(LogTemp, Error, TEXT("TRIAD RL training reset failed: %s"), *Error);
        Destroy();
        return;
    }
    UE_LOG(LogTemp, Display, TEXT("TRIAD simulation-only RL environment ready in Blue placement phase."));
}

void ATRIADAdversarialTrainingManager::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    ClearEpisodeActors();
    Super::EndPlay(EndPlayReason);
}

bool ATRIADAdversarialTrainingManager::ResetEpisode(
    int32 Seed, FTRIADRLStepResult& OutResult, FString& OutError)
{
    if (!Georeference)
    {
        Georeference = FindGeoreference();
    }
    if (!Georeference || !FTRIADRLTrainingModel::ValidateConfig(TrainingConfig, OutError))
    {
        return false;
    }

    ClearEpisodeActors();
    Phase = ETRIADRLPhase::BluePlacement;
    TerminationReason = ETRIADRLTerminationReason::None;
    StepIndex = 0;
    TransitionIndex = 0;
    LastDetectedTargetCount = 0;
    CurrentlyDetectedTargetCount = 0;
    TrackedTargetCount = 0;
    AllTargetsTrackedSteps = 0;
    InvalidActionCount = 0;
    LastStepRewards = FTRIADRLRewardBreakdown();
    LastBlueAction.Reset();
    LastRedDeploymentAction.Reset();
    LastRedMovementActions.Reset();
    SpentSensorBudgetUnits = 0.0;
    AccumulatedBlueReward = 0.0;
    AccumulatedRedReward = 0.0;
    EpisodeSeed = Seed;
    if (!ResolveMapObjective(OutError))
    {
        Phase = ETRIADRLPhase::Terminal;
        TerminationReason = ETRIADRLTerminationReason::ConstraintViolation;
        return false;
    }
    PreviousMinimumZoneDistanceMeters = TrainingConfig.MaximumDistanceFromZoneMeters;
    if (!ResolveSensorCatalogue(OutError)) return false;
    BuildStepResult(OutResult);
    OutError.Reset();
    return true;
}

bool ATRIADAdversarialTrainingManager::ApplyBlueAction(
    const FTRIADRLBlueAction& Action, FTRIADRLStepResult& OutResult, FString& OutError)
{
    if (Phase != ETRIADRLPhase::BluePlacement)
    {
        ++InvalidActionCount;
        OutError = TEXT("Blue actions are accepted only during the placement phase.");
        return false;
    }
    if (Action.bStopPlacement)
    {
        LastStepRewards = FTRIADRLRewardBreakdown();
        LastBlueAction = TEXT("Commit layout");
        Phase = ETRIADRLPhase::RedDeployment;
        ++TransitionIndex;
        BuildStepResult(OutResult);
        OutError.Reset();
        return true;
    }
    if (!SpawnSensorFromCatalogue(Action.CatalogueIndex, Action.NormalizedPosition, OutError))
    {
        ++InvalidActionCount;
        BuildStepResult(OutResult);
        return false;
    }
    LastStepRewards = FTRIADRLRewardBreakdown();
    LastStepRewards.BlueEfficiency = TrainingConfig.Rewards.BlueSiteCost *
        TrainingConfig.SensorCandidates[Action.CatalogueIndex].CostUnits / FMath::Max(TrainingConfig.SensorBudgetUnits, 0.000001);
    AccumulatedBlueReward += LastStepRewards.BlueTotal();
    const auto& SelectedOption = TrainingConfig.SensorCandidates[Action.CatalogueIndex];
    const FVector2D SelectedPosition = SelectedOption.bAllowDynamicPosition
        ? Action.NormalizedPosition
        : FVector2D(SelectedOption.OffsetEnuMeters.X / TrainingConfig.BluePlacementRadiusMeters,
            SelectedOption.OffsetEnuMeters.Y / TrainingConfig.BluePlacementRadiusMeters);
    LastBlueAction = FString::Printf(TEXT("%s [%s] E %.3f N %.3f"),
        *SelectedOption.CandidateId, *SelectedOption.SensorProfileId, SelectedPosition.X, SelectedPosition.Y);
    ++TransitionIndex;
    if (SpawnedSensors.Num() >= TrainingConfig.MaximumSensorSites)
    {
        Phase = ETRIADRLPhase::RedDeployment;
    }
    BuildStepResult(OutResult);
    OutError.Reset();
    return true;
}

bool ATRIADAdversarialTrainingManager::ApplyRedDeploymentAction(
    const FTRIADRLRedDeploymentAction& Action,
    FTRIADRLStepResult& OutResult,
    FString& OutError)
{
    if (Phase != ETRIADRLPhase::RedDeployment)
    {
        ++InvalidActionCount;
        OutError = TEXT("Red deployment is accepted only after Blue commits its layout.");
        return false;
    }
    if (!SpawnTargetsFromDeployment(Action, OutError))
    {
        ++InvalidActionCount;
        BuildStepResult(OutResult);
        return false;
    }
    PreviousMinimumZoneDistanceMeters = ComputeMinimumZoneDistanceMeters();
    LastStepRewards = FTRIADRLRewardBreakdown();
    LastRedDeploymentAction = FString::Printf(TEXT("bearing %.2f radius %.2f altitude %.2f count %.2f spacing %.2f"),
        Action.NormalizedBearing, Action.NormalizedRadius, Action.NormalizedAltitude,
        Action.NormalizedSwarmSize, Action.NormalizedFormationSpacing);
    Phase = ETRIADRLPhase::RedMovement;
    ++TransitionIndex;
    BuildStepResult(OutResult);
    OutError.Reset();
    return true;
}

bool ATRIADAdversarialTrainingManager::ApplyRedActions(
    const TArray<FTRIADRLRedAction>& Actions, FTRIADRLStepResult& OutResult, FString& OutError)
{
    if (Phase != ETRIADRLPhase::RedMovement)
    {
        ++InvalidActionCount;
        OutError = TEXT("Red actions are accepted only during the movement phase.");
        return false;
    }
    if (Actions.Num() != TargetControllers.Num())
    {
        ++InvalidActionCount;
        OutError = FString::Printf(
            TEXT("Red action count %d does not match active synthetic target count %d."),
            Actions.Num(), TargetControllers.Num());
        return false;
    }

    // Validate the WHOLE action batch before any target moves. Rejected input
    // cannot leave the first few targets one transition ahead of the others.
    for (int32 Index = 0; Index < Actions.Num(); ++Index)
    {
        const FVector& V = Actions[Index].NormalizedVelocityEnu;
        if (!IsValid(TargetControllers[Index]) || !FMath::IsFinite(V.X) || !FMath::IsFinite(V.Y) ||
            !FMath::IsFinite(V.Z) || V.GetAbsMax() > 1.0)
        {
            ++InvalidActionCount;
            OutError = TEXT("Red movement must contain a finite normalized vector for every live target.");
            BuildStepResult(OutResult);
            return false;
        }
    }
    bool bMotionConstraintViolation = false;
    LastRedMovementActions.Reset();
    for (int32 Index = 0; Index < Actions.Num(); ++Index)
    {
        const FVector ProjectedAction = Actions[Index].NormalizedVelocityEnu * TrainingConfig.RedMovementAxisMask;
        LastRedMovementActions.Add(ProjectedAction);
        if (!TargetControllers[Index]->ApplyNormalizedVelocity(ProjectedAction, TrainingConfig.FixedStepSeconds))
        {
            bMotionConstraintViolation = true;
        }
        TargetPaths[Index].Add(SpawnedTargets[Index]->GetActorLocation());
    }
    ++StepIndex;
    ++TransitionIndex;
    const double CurrentMinimumDistance = ComputeMinimumZoneDistanceMeters();
    const int32 NewlyDetectedTargets = EvaluateDetectedTargets();
    LastDetectedTargetCount += NewlyDetectedTargets;
    const bool bZoneReached = HasProtectedZoneEntry();
    const bool bConstraintViolation = bMotionConstraintViolation || HasConstraintViolation();
    const bool bAllTracked = !TargetTracks.IsEmpty() && TrackedTargetCount == TargetTracks.Num();
    AllTargetsTrackedSteps = bAllTracked ? AllTargetsTrackedSteps + 1 : 0;

    if (bZoneReached)
    {
        Phase = ETRIADRLPhase::Terminal;
        TerminationReason = ETRIADRLTerminationReason::ProtectedZoneReached;
    }
    else if (bConstraintViolation)
    {
        Phase = ETRIADRLPhase::Terminal;
        TerminationReason = ETRIADRLTerminationReason::ConstraintViolation;
    }
    else if (AllTargetsTrackedSteps >= TrainingConfig.DefenceTrackHoldSteps)
    {
        Phase = ETRIADRLPhase::Terminal;
        TerminationReason = ETRIADRLTerminationReason::SustainedTrackDefence;
    }
    else if (StepIndex >= TrainingConfig.EpisodeHorizonSteps)
    {
        Phase = ETRIADRLPhase::Terminal;
        TerminationReason = ETRIADRLTerminationReason::HorizonReached;
    }

    const double TargetCount = FMath::Max(SpawnedTargets.Num(), 1);
    FTRIADRLTrainingModel::ComputeStepRewards(TrainingConfig,
        PreviousMinimumZoneDistanceMeters, CurrentMinimumDistance,
        NewlyDetectedTargets / TargetCount,
        1.0 - static_cast<double>(StepIndex) / TrainingConfig.EpisodeHorizonSteps,
        TrackedTargetCount / TargetCount, TerminationReason, LastStepRewards);
    AccumulatedBlueReward += LastStepRewards.BlueTotal();
    AccumulatedRedReward += LastStepRewards.RedTotal();
    PreviousMinimumZoneDistanceMeters = CurrentMinimumDistance;

    BuildStepResult(OutResult);
    if (OutResult.bTerminal && bEvaluationOnly) LogTerminalEvaluation(OutResult);
    OutError.Reset();
    return true;
}

void ATRIADAdversarialTrainingManager::ClearEpisodeActors()
{
    for (ATRIADSensorNodeActor* Sensor : SpawnedSensors)
    {
        if (IsValid(Sensor)) Sensor->Destroy();
    }
    for (ATRIADDemoDroneActor* Target : SpawnedTargets)
    {
        if (IsValid(Target)) Target->Destroy();
    }
    SpawnedSensors.Reset();
    SpawnedTargets.Reset();
    TargetControllers.Reset();
    PlacedCatalogueIndices.Reset();
    TargetTracks.Reset();
    TargetPaths.Reset();
    SensorDetectionCounts.Reset();
    CatalogueDetectionCounts.Reset();
}

bool ATRIADAdversarialTrainingManager::ResolveMapObjective(FString& OutError)
{
    ObjectiveActor = nullptr;
    int32 MatchCount = 0;
    if (GetWorld())
    {
        for (TActorIterator<AActor> It(GetWorld()); It; ++It)
        {
            if (IsValid(*It) && It->ActorHasTag(TEXT("TRIADRLObjective")))
            {
                ObjectiveActor = *It;
                ++MatchCount;
            }
        }
    }
    if (MatchCount != 1 || !ObjectiveActor || !Georeference)
    {
        OutError = FString::Printf(
            TEXT("RL map requires exactly one actor tagged TRIADRLObjective; found %d."), MatchCount);
        ObjectiveActor = nullptr;
        return false;
    }
    const FVector ObjectiveLongitudeLatitudeHeight =
        Georeference->TransformUnrealPositionToLongitudeLatitudeHeight(ObjectiveActor->GetActorLocation());
    TrainingConfig.ProtectedZone.LongitudeDegrees = ObjectiveLongitudeLatitudeHeight.X;
    TrainingConfig.ProtectedZone.LatitudeDegrees = ObjectiveLongitudeLatitudeHeight.Y;
    TrainingConfig.ProtectedZone.HeightMeters = ObjectiveLongitudeLatitudeHeight.Z;
    ProtectedZone->Configure(TrainingConfig.ProtectedZone);
    return true;
}

bool ATRIADAdversarialTrainingManager::SpawnTargetsFromDeployment(
    const FTRIADRLRedDeploymentAction& Action, FString& OutError)
{
    const double Values[] = {
        Action.NormalizedBearing,
        Action.NormalizedRadius,
        Action.NormalizedAltitude,
        Action.NormalizedSwarmSize,
        Action.NormalizedFormationSpacing};
    for (const double Value : Values)
    {
        if (!FMath::IsFinite(Value) || Value < -1.0 || Value > 1.0)
        {
            OutError = TEXT("Red deployment values must be finite and normalized to [-1, 1].");
            return false;
        }
    }
    if (!GetWorld() || !ObjectiveActor)
    {
        OutError = TEXT("RL world or objective is unavailable for Red deployment.");
        return false;
    }

    const auto UnitInterval = [](double Value) { return (Value + 1.0) * 0.5; };
    const double BearingRadians = FMath::DegreesToRadians(FMath::Lerp(
        TrainingConfig.RedMinimumBearingDegrees, TrainingConfig.RedMaximumBearingDegrees,
        UnitInterval(Action.NormalizedBearing)));
    const double RadiusMeters = FMath::Lerp(
        TrainingConfig.RedMinimumSpawnRadiusMeters,
        TrainingConfig.RedMaximumSpawnRadiusMeters,
        UnitInterval(Action.NormalizedRadius));
    const double AltitudeMeters = FMath::Lerp(
        TrainingConfig.RedMinimumAltitudeMeters,
        TrainingConfig.RedMaximumAltitudeMeters,
        UnitInterval(Action.NormalizedAltitude));
    const int32 Count = FMath::Clamp(
        FMath::RoundToInt(FMath::Lerp(
            static_cast<double>(TrainingConfig.RedMinimumSwarmSize),
            static_cast<double>(TrainingConfig.RedMaximumSwarmSize),
            UnitInterval(Action.NormalizedSwarmSize))),
        1,
        MaximumRLTargetActors);
    const double FormationSpacingMeters = FMath::Lerp(
        TrainingConfig.RedMinimumFormationSpacingMeters,
        TrainingConfig.RedMaximumFormationSpacingMeters,
        UnitInterval(Action.NormalizedFormationSpacing));
    const double RadialEastMeters = FMath::Sin(BearingRadians) * RadiusMeters;
    const double RadialNorthMeters = FMath::Cos(BearingRadians) * RadiusMeters;
    const FVector2D Tangent(FMath::Cos(BearingRadians), -FMath::Sin(BearingRadians));
    constexpr double EarthRadiusMeters = 6378137.0;
    for (int32 Index = 0; Index < Count; ++Index)
    {
        const double FormationOffset = (Index - (Count - 1) * 0.5) * FormationSpacingMeters;
        const double EastOffset = RadialEastMeters + Tangent.X * FormationOffset;
        const double NorthOffset = RadialNorthMeters + Tangent.Y * FormationOffset;
        const double CenterLatitude = TrainingConfig.ProtectedZone.LatitudeDegrees;
        const double CosLatitude = FMath::Max(
            FMath::Abs(FMath::Cos(FMath::DegreesToRadians(CenterLatitude))), 0.000001);
        FTRIADDemoTargetDefinition Definition;
        Definition.ActorName = FString::Printf(TEXT("RL_Target_%02d"), Index + 1);
        Definition.StartLongitudeDegrees = TrainingConfig.ProtectedZone.LongitudeDegrees +
            FMath::RadiansToDegrees(EastOffset / (EarthRadiusMeters * CosLatitude));
        Definition.StartLatitudeDegrees = CenterLatitude +
            FMath::RadiansToDegrees(NorthOffset / EarthRadiusMeters);
        Definition.StartHeightMeters = TrainingConfig.ProtectedZone.HeightMeters + AltitudeMeters;
        Definition.Trajectory = ETRIADDemoTrajectory::Stationary;
        Definition.RFEmitter.EmitterId = Definition.ActorName + TEXT("_SyntheticRF");
        Definition.RFEmitter.CenterFrequenciesGHz = {2.437, 5.795};
        Definition.RFEmitter.TransmitPowerDbm = 20.0;
        Definition.RFEmitter.TransmitAntennaGainDbi = 2.0;
        Definition.RFEmitter.bEnabled = true;

        FActorSpawnParameters Parameters;
        Parameters.Name = MakeUniqueObjectName(GetWorld(), ATRIADDemoDroneActor::StaticClass(), FName(*Definition.ActorName));
        Parameters.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
        ATRIADDemoDroneActor* Target = GetWorld()->SpawnActor<ATRIADDemoDroneActor>(
            ATRIADDemoDroneActor::StaticClass(), FTransform::Identity, Parameters);
        if (!Target)
        {
            for (ATRIADDemoDroneActor* Created : SpawnedTargets) if (IsValid(Created)) Created->Destroy();
            SpawnedTargets.Reset();
            TargetControllers.Reset();
            TargetTracks.Reset();
            TargetPaths.Reset();
            OutError = TEXT("Could not spawn an RL synthetic target.");
            return false;
        }
        Target->ConfigureDemoTarget(Definition, Georeference);
        UTRIADSwarmControllerComponent* Controller = NewObject<UTRIADSwarmControllerComponent>(Target);
        Controller->RegisterComponent();
        Controller->Configure(
            Georeference,
            Definition.StartLongitudeDegrees,
            Definition.StartLatitudeDegrees,
            Definition.StartHeightMeters,
            TrainingConfig.MaximumTargetSpeedMetersPerSecond);
        SpawnedTargets.Add(Target);
        TargetControllers.Add(Controller);
        TargetTracks.Add(FTRIADRLTrackState());
        TargetPaths.Add(TArray<FVector>{Target->GetActorLocation()});
    }
    return true;
}

bool ATRIADAdversarialTrainingManager::ResolveSensorCatalogue(FString& OutError)
{
    ResolvedMounts.Reset();
    CandidateWorldPositions.Reset();
    constexpr double EarthRadius = 6378137.0;
    const double CosLat = FMath::Max(FMath::Abs(FMath::Cos(
        FMath::DegreesToRadians(TrainingConfig.ProtectedZone.LatitudeDegrees))), 0.000001);
    for (int32 Index = 0; Index < TrainingConfig.SensorCandidates.Num(); ++Index)
    {
        const auto& Option = TrainingConfig.SensorCandidates[Index];
        const double Longitude = TrainingConfig.ProtectedZone.LongitudeDegrees +
            FMath::RadiansToDegrees(Option.OffsetEnuMeters.X / (EarthRadius * CosLat));
        const double Latitude = TrainingConfig.ProtectedZone.LatitudeDegrees +
            FMath::RadiansToDegrees(Option.OffsetEnuMeters.Y / EarthRadius);
        FVector Position = Georeference->TransformLongitudeLatitudeHeightPositionToUnreal(FVector(
            Longitude, Latitude, TrainingConfig.ProtectedZone.HeightMeters + Option.OffsetEnuMeters.Z));
        FTRIADRLMountObservation Mount;
        Mount.CatalogueIndex = Index;
        Mount.CandidateId = Option.CandidateId;
        Mount.SensorProfileId = Option.SensorProfileId;
        Mount.MountId = Option.MountId;
        Mount.OffsetEnuMeters = Option.OffsetEnuMeters;
        Mount.YawDegrees = Option.YawDegrees;
        Mount.PitchDegrees = Option.PitchDegrees;
        Mount.HorizontalFovDegrees = Option.HorizontalFovDegrees;
        Mount.VerticalFovDegrees = Option.VerticalFovDegrees;
        Mount.CostUnits = Option.CostUnits;
        Mount.SensorModalityMask = FTRIADRLTrainingModel::GetSensorModalityMask(Option.Sensor);
        Mount.bAllowDynamicPosition = Option.bAllowDynamicPosition;
        Mount.RangeMeters = FMath::Max(
            FMath::Max(Option.Sensor.DetectionRangeMeters,
                Option.Sensor.bEnableSearchRadar ? Option.Sensor.SearchRadarRangeMeters : 0.0),
            FMath::Max(Option.Sensor.bEnableEOPTZ ? Option.Sensor.EOPTZConfirmationRangeMeters : 0.0,
                Option.Sensor.bEnableThermalPTZ ? Option.Sensor.ThermalPTZConfirmationRangeMeters : 0.0));
        Mount.bEnabled = Option.bApproved && Option.Sensor.bEnabled;
        // A dynamic profile resolves the exact surface when its continuous action is accepted.
        Mount.bSurfaceResolved = Option.bAllowDynamicPosition || !Option.bProjectToGround;
        if (Option.bProjectToGround && !Option.bAllowDynamicPosition)
        {
            const FVector Start = Georeference->TransformLongitudeLatitudeHeightPositionToUnreal(
                FVector(Longitude, Latitude, TrainingConfig.ProtectedZone.HeightMeters + 1500.0));
            const FVector End = Georeference->TransformLongitudeLatitudeHeightPositionToUnreal(
                FVector(Longitude, Latitude, TrainingConfig.ProtectedZone.HeightMeters - 500.0));
            FCollisionQueryParams Params(SCENE_QUERY_STAT(TRIADRLCatalogueGround), true);
            FHitResult Hit;
            Mount.bSurfaceResolved = GetWorld()->LineTraceSingleByChannel(Hit, Start, End, ECC_Visibility, Params);
            if (Mount.bSurfaceResolved)
            {
                FVector LLH = Georeference->TransformUnrealPositionToLongitudeLatitudeHeight(Hit.ImpactPoint);
                LLH.Z += Option.OffsetEnuMeters.Z;
                Position = Georeference->TransformLongitudeLatitudeHeightPositionToUnreal(LLH);
                Mount.OffsetEnuMeters.Z = LLH.Z - TrainingConfig.ProtectedZone.HeightMeters;
                const FVector ESU = Georeference->ComputeUnrealToEastSouthUpTransformation(Hit.ImpactPoint).TransformVector(Hit.ImpactNormal);
                Mount.SurfaceNormal = FVector(ESU.X, -ESU.Y, ESU.Z).GetSafeNormal();
            }
        }
        ResolvedMounts.Add(Mount);
        CandidateWorldPositions.Add(Position);
    }
    OutError.Reset();
    return true;
}

bool ATRIADAdversarialTrainingManager::CanPlaceCandidate(int32 Index) const
{
    if (Phase != ETRIADRLPhase::BluePlacement || !ResolvedMounts.IsValidIndex(Index) ||
        !CandidateWorldPositions.IsValidIndex(Index) || SpawnedSensors.Num() >= TrainingConfig.MaximumSensorSites)
        return false;
    const auto& Mount = ResolvedMounts[Index];
    const auto& Option = TrainingConfig.SensorCandidates[Index];
    if (!Mount.bSurfaceResolved || !Mount.bEnabled ||
        SpentSensorBudgetUnits + Mount.CostUnits > TrainingConfig.SensorBudgetUnits + UE_DOUBLE_SMALL_NUMBER)
        return false;
    for (int32 PlacedSlot = 0; PlacedSlot < PlacedCatalogueIndices.Num(); ++PlacedSlot)
    {
        const int32 PlacedIndex = PlacedCatalogueIndices[PlacedSlot];
        const auto& PlacedOption = TrainingConfig.SensorCandidates[PlacedIndex];
        if (!Option.bAllowDynamicPosition && !PlacedOption.bAllowDynamicPosition &&
            ResolvedMounts[PlacedIndex].MountId == Mount.MountId) return false;
        if (!Option.bAllowDynamicPosition && SpawnedSensors.IsValidIndex(PlacedSlot) && SpawnedSensors[PlacedSlot] &&
            FVector::Dist2D(SpawnedSensors[PlacedSlot]->GetActorLocation(), CandidateWorldPositions[Index]) * 0.01 <
                TrainingConfig.MinimumSensorSeparationMeters) return false;
    }
    return true;
}

bool ATRIADAdversarialTrainingManager::SpawnSensorFromCatalogue(
    int32 Index, const FVector2D& RequestedNormalizedPosition, FString& OutError)
{
    if (!CanPlaceCandidate(Index))
    {
        OutError = TEXT("Blue catalogue index is masked: unavailable mount, duplicate, separation, site limit or budget.");
        return false;
    }
    const auto& Option = TrainingConfig.SensorCandidates[Index];
    const FVector2D NormalizedPosition = Option.bAllowDynamicPosition
        ? RequestedNormalizedPosition
        : FVector2D(Option.OffsetEnuMeters.X / TrainingConfig.BluePlacementRadiusMeters,
            Option.OffsetEnuMeters.Y / TrainingConfig.BluePlacementRadiusMeters);
    if (!FMath::IsFinite(NormalizedPosition.X) || !FMath::IsFinite(NormalizedPosition.Y) ||
        NormalizedPosition.SizeSquared() > 1.0 + UE_DOUBLE_SMALL_NUMBER)
    {
        OutError = TEXT("Blue placement must be a finite point inside the normalized unit disk.");
        return false;
    }
    const FVector2D OffsetMeters = NormalizedPosition * TrainingConfig.BluePlacementRadiusMeters;
    if (OffsetMeters.Length() < TrainingConfig.BlueMinimumObjectiveStandoffMeters)
    {
        OutError = TEXT("Blue placement is inside the protected objective standoff.");
        return false;
    }
    if (SpawnedSensors.Num() >= TrainingConfig.MaximumSensorSites ||
        SpentSensorBudgetUnits + Option.CostUnits > TrainingConfig.SensorBudgetUnits + UE_DOUBLE_SMALL_NUMBER)
    {
        OutError = TEXT("Blue sensor-site or budget limit would be exceeded.");
        return false;
    }

    constexpr double EarthRadiusMeters = 6378137.0;
    const double CenterLatitudeRadians = FMath::DegreesToRadians(TrainingConfig.ProtectedZone.LatitudeDegrees);
    const double CosLatitude = FMath::Max(FMath::Abs(FMath::Cos(CenterLatitudeRadians)), 0.000001);
    const double CandidateLongitude = TrainingConfig.ProtectedZone.LongitudeDegrees +
        FMath::RadiansToDegrees(OffsetMeters.X / (EarthRadiusMeters * CosLatitude));
    const double CandidateLatitude = TrainingConfig.ProtectedZone.LatitudeDegrees +
        FMath::RadiansToDegrees(OffsetMeters.Y / EarthRadiusMeters);
    FVector MountPosition = CandidateWorldPositions[Index];
    if (Option.bAllowDynamicPosition)
    {
        FVector PlacementLongitudeLatitudeHeight(
            CandidateLongitude,
            CandidateLatitude,
            TrainingConfig.ProtectedZone.HeightMeters + Option.OffsetEnuMeters.Z);
        if (Option.bProjectToGround)
        {
            const FVector Start = Georeference->TransformLongitudeLatitudeHeightPositionToUnreal(FVector(
                CandidateLongitude, CandidateLatitude, TrainingConfig.ProtectedZone.HeightMeters + 1500.0));
            const FVector End = Georeference->TransformLongitudeLatitudeHeightPositionToUnreal(FVector(
                CandidateLongitude, CandidateLatitude, TrainingConfig.ProtectedZone.HeightMeters - 500.0));
            FCollisionQueryParams Params(SCENE_QUERY_STAT(TRIADRLContinuousGround), true);
            FHitResult Hit;
            if (!GetWorld()->LineTraceSingleByChannel(Hit, Start, End, ECC_Visibility, Params))
            {
                OutError = TEXT("Dynamic Blue placement did not resolve collision-supported ground.");
                return false;
            }
            PlacementLongitudeLatitudeHeight = Georeference->TransformUnrealPositionToLongitudeLatitudeHeight(Hit.ImpactPoint);
            PlacementLongitudeLatitudeHeight.Z += Option.OffsetEnuMeters.Z;
        }
        MountPosition = Georeference->TransformLongitudeLatitudeHeightPositionToUnreal(PlacementLongitudeLatitudeHeight);
    }
    for (const ATRIADSensorNodeActor* ExistingSensor : SpawnedSensors)
    {
        if (ExistingSensor && FVector::Dist2D(ExistingSensor->GetActorLocation(), MountPosition) / 100.0 <
                TrainingConfig.MinimumSensorSeparationMeters)
        {
            OutError = TEXT("Blue placement violates minimum sensor separation.");
            return false;
        }
    }
    const FVector SurfaceLongitudeLatitudeHeight =
        Georeference->TransformUnrealPositionToLongitudeLatitudeHeight(MountPosition);
    FTRIADGeodeticSensorNode SensorDefinition = Option.Sensor;
    SensorDefinition.NodeId = FString::Printf(TEXT("RL_Blue_%02d"), SpawnedSensors.Num() + 1);
    SensorDefinition.LongitudeDegrees = SurfaceLongitudeLatitudeHeight.X;
    SensorDefinition.LatitudeDegrees = SurfaceLongitudeLatitudeHeight.Y;
    SensorDefinition.HeightMeters = SurfaceLongitudeLatitudeHeight.Z;
    SensorDefinition.bEnabled = true;
    SensorDefinition.bCaptureCameraFrames = false;
    SensorDefinition.bTrackNearestTarget = false;
    // RL draws an exact catalogue-specific overlay. The ordinary scenario's
    // legacy node label describes its separate long-range/PTZ presentation.
    SensorDefinition.bVisualizeNode = false;
    FActorSpawnParameters Parameters;
    Parameters.Name = MakeUniqueObjectName(GetWorld(), ATRIADSensorNodeActor::StaticClass(),
        FName(*FString::Printf(TEXT("TRIAD_RL_Sensor_%02d"), SpawnedSensors.Num() + 1)));
    Parameters.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
    ATRIADSensorNodeActor* Sensor = GetWorld()->SpawnActor<ATRIADSensorNodeActor>(
        ATRIADSensorNodeActor::StaticClass(), FTransform::Identity, Parameters);
    if (!Sensor)
    {
        OutError = TEXT("Could not spawn a Blue sensor at the approved catalogue mount.");
        return false;
    }
    Sensor->ConfigureNode(
        SensorDefinition,
        Georeference,
        FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("TRIAD"), TEXT("RL"), TEXT("frames")),
        0,
        0);
    Sensor->SetActorTickEnabled(false); // manager owns pose/FOV visualization, not the demo PTZ loop
    Sensor->SetActorRotation(Georeference->TransformEastSouthUpRotatorToUnreal(
        FRotator(Option.PitchDegrees, Option.YawDegrees - 90.0, 0.0), MountPosition));
    SpawnedSensors.Add(Sensor);
    SensorDetectionCounts.Add(0);
    PlacedCatalogueIndices.Add(Index);
    SpentSensorBudgetUnits += Option.CostUnits;
    return true;
}

int32 ATRIADAdversarialTrainingManager::EvaluateDetectedTargets()
{
    int32 NewlyDetectedTargets = 0;
    CurrentlyDetectedTargetCount = 0;
    TrackedTargetCount = 0;
    for (int32& Count : SensorDetectionCounts) Count = 0;
    CatalogueDetectionCounts.Init(0, TrainingConfig.SensorCandidates.Num());
    for (int32 TargetIndex = 0; TargetIndex < SpawnedTargets.Num(); ++TargetIndex)
    {
        ATRIADDemoDroneActor* Target = SpawnedTargets[TargetIndex];
        if (!IsValid(Target) || !Target->RFEmitter)
        {
            FTRIADRLTrainingModel::AdvanceTrack(TargetTracks[TargetIndex], false, StepIndex, TrainingConfig.TrackConfirmationSteps);
            continue;
        }
        int32 ConfirmingNodes = 0;
        TSet<int32> DetectingCatalogueIndices;
        for (int32 SensorIndex = 0; SensorIndex < SpawnedSensors.Num(); ++SensorIndex)
        {
            ATRIADSensorNodeActor* Sensor = SpawnedSensors[SensorIndex];
            if (!IsValid(Sensor) || !Sensor->SensorNode) continue;
            const double DistanceMeters = FVector::Distance(Sensor->GetActorLocation(), Target->GetActorLocation()) / 100.0;
            const FTRIADGeodeticSensorNode& Node = Sensor->GetNodeDefinition();
            const auto& Pose = TrainingConfig.SensorCandidates[PlacedCatalogueIndices[SensorIndex]];
            const FVector TargetLLH = Georeference->TransformUnrealPositionToLongitudeLatitudeHeight(Target->GetActorLocation());
            constexpr double EarthRadius = 6378137.0;
            const double East = FMath::DegreesToRadians(TargetLLH.X - Node.LongitudeDegrees) * EarthRadius *
                FMath::Cos(FMath::DegreesToRadians(Node.LatitudeDegrees));
            const double North = FMath::DegreesToRadians(TargetLLH.Y - Node.LatitudeDegrees) * EarthRadius;
            const double Bearing = FMath::RadiansToDegrees(FMath::Atan2(East, North));
            const double Elevation = FMath::RadiansToDegrees(FMath::Atan2(TargetLLH.Z - Node.HeightMeters,
                FMath::Sqrt(East * East + North * North)));
            const bool bInsideDirectionalFieldOfView =
                FMath::Abs(FMath::FindDeltaAngleDegrees(Pose.YawDegrees, Bearing)) <= Pose.HorizontalFovDegrees * 0.5 &&
                FMath::Abs(Elevation - Pose.PitchDegrees) <= Pose.VerticalFovDegrees * 0.5;
            FCollisionQueryParams QueryParams(SCENE_QUERY_STAT(TRIADRLSensorLOS), true);
            QueryParams.AddIgnoredActor(Sensor);
            QueryParams.AddIgnoredActor(Target);
            FHitResult Hit;
            const bool bLineOfSight = !GetWorld()->LineTraceSingleByChannel(
                Hit, Sensor->GetActorLocation(), Target->GetActorLocation(), ECC_Visibility, QueryParams);

            bool bNodeDetected = false;
            if (Target->RFEmitter->IsEmitting() && DistanceMeters <= FMath::Max(Node.DetectionRangeMeters, 0.0))
            {
                for (const double FrequencyGHz : Target->RFEmitter->GetCenterFrequenciesGHz())
                {
                    if (!Sensor->SensorNode->SupportsFrequencyGHz(FrequencyGHz)) continue;
                    const double ReceivedPowerDbm = Target->RFEmitter->Definition.TransmitPowerDbm +
                        Target->RFEmitter->Definition.TransmitAntennaGainDbi + Node.ReceiveAntennaGainDbi -
                        Sensor->SensorNode->ComputeFreeSpacePathLossDb(FrequencyGHz, DistanceMeters) - Node.SystemLossDb -
                        (bLineOfSight ? 0.0 : 30.0);
                    if (ReceivedPowerDbm >= Node.ReceiverSensitivityDbm)
                    {
                        bNodeDetected = true;
                        break;
                    }
                }
            }
            if (!bNodeDetected && Node.bEnableSearchRadar)
            {
                FTRIADSearchRadarModelInput Input;
                Input.TrueRangeMeters = DistanceMeters;
                Input.RadarCrossSectionSquareMeters = Target->GetSimulatedRadarCrossSectionSquareMeters();
                Input.RangeEnvelopeMeters = Node.SearchRadarRangeMeters;
                Input.ElevationFieldOfRegardDegrees = Node.SearchRadarElevationFieldOfRegardDegrees;
                Input.DetectionThreshold = Node.SearchRadarDetectionThreshold;
                Input.bLineOfSight = bLineOfSight;
                Input.TrueBearingDegrees = Bearing;
                Input.TrueElevationDegrees = Elevation - Pose.PitchDegrees;
                Input.DeterministicSeed = HashCombine(GetTypeHash(EpisodeSeed), HashCombine(GetTypeHash(StepIndex),
                    HashCombine(GetTypeHash(TargetIndex), GetTypeHash(SensorIndex))));
                bNodeDetected = FTRIADLongRangeSensorModel::EvaluateSearchRadar(Input).bDetected;
            }
            // Optical profiles use deterministic analytic confirmation in the RL
            // loop. Frame capture remains disabled, so training cannot depend on
            // render timing or leak image-side state into the transition model.
            if (!bNodeDetected && Node.bEnableEOPTZ && bInsideDirectionalFieldOfView && bLineOfSight &&
                DistanceMeters <= Node.EOPTZConfirmationRangeMeters)
            {
                bNodeDetected = true;
            }
            if (!bNodeDetected && Node.bEnableThermalPTZ && bInsideDirectionalFieldOfView && bLineOfSight &&
                DistanceMeters <= Node.ThermalPTZConfirmationRangeMeters)
            {
                bNodeDetected = true;
            }
            ConfirmingNodes += bNodeDetected ? 1 : 0;
            SensorDetectionCounts[SensorIndex] += bNodeDetected ? 1 : 0;
            if (bNodeDetected)
            {
                DetectingCatalogueIndices.Add(PlacedCatalogueIndices[SensorIndex]);
            }
        }
        for (const int32 CatalogueIndex : DetectingCatalogueIndices)
        {
            ++CatalogueDetectionCounts[CatalogueIndex];
        }
        const bool bDetected = ConfirmingNodes >= TrainingConfig.ConfirmationNodeCount;
        NewlyDetectedTargets += FTRIADRLTrainingModel::AdvanceTrack(TargetTracks[TargetIndex], bDetected,
            StepIndex, TrainingConfig.TrackConfirmationSteps) ? 1 : 0;
        CurrentlyDetectedTargetCount += bDetected ? 1 : 0;
        TrackedTargetCount += TargetTracks[TargetIndex].bTracked ? 1 : 0;
    }
    return NewlyDetectedTargets;
}

double ATRIADAdversarialTrainingManager::ComputeMinimumZoneDistanceMeters() const
{
    double Minimum = TNumericLimits<double>::Max();
    for (const UTRIADSwarmControllerComponent* Controller : TargetControllers)
    {
        if (Controller)
        {
            Minimum = FMath::Min(Minimum, ProtectedZone->SignedHorizontalDistanceMeters(
                Controller->GetLongitudeDegrees(), Controller->GetLatitudeDegrees()));
        }
    }
    return Minimum == TNumericLimits<double>::Max()
        ? TrainingConfig.MaximumDistanceFromZoneMeters
        : Minimum;
}

bool ATRIADAdversarialTrainingManager::HasConstraintViolation() const
{
    for (const UTRIADSwarmControllerComponent* Controller : TargetControllers)
    {
        if (!Controller || ProtectedZone->SignedHorizontalDistanceMeters(
                Controller->GetLongitudeDegrees(), Controller->GetLatitudeDegrees()) >
                TrainingConfig.MaximumDistanceFromZoneMeters)
        {
            return true;
        }
    }
    return false;
}

bool ATRIADAdversarialTrainingManager::HasProtectedZoneEntry() const
{
    return !TargetControllers.IsEmpty() && ComputeMinimumZoneDistanceMeters() <= 0.0;
}

void ATRIADAdversarialTrainingManager::BuildStepResult(FTRIADRLStepResult& OutResult) const
{
    OutResult = FTRIADRLStepResult();
    OutResult.ScenarioId = TrainingConfig.ScenarioId;
    OutResult.ConfigFingerprint = ConfigFingerprint;
    OutResult.EpisodeSeed = EpisodeSeed;
    OutResult.TransitionIndex = TransitionIndex;
    OutResult.RemainingBudgetUnits = TrainingConfig.SensorBudgetUnits - SpentSensorBudgetUnits;
    OutResult.RemainingSiteCount = TrainingConfig.MaximumSensorSites - SpawnedSensors.Num();
    OutResult.MountObservations = ResolvedMounts;
    OutResult.PlacedCatalogueIndices = PlacedCatalogueIndices;
    for (int32 Index = 0; Index < OutResult.MountObservations.Num(); ++Index)
    {
        OutResult.MountObservations[Index].CurrentlyDetectingTargetCount =
            CatalogueDetectionCounts.IsValidIndex(Index) ? CatalogueDetectionCounts[Index] : 0;
    }
    for (auto& Mount : OutResult.MountObservations)
        for (int32 PlacedIndex : PlacedCatalogueIndices)
            Mount.bOccupied |= !Mount.bAllowDynamicPosition &&
                !ResolvedMounts[PlacedIndex].bAllowDynamicPosition &&
                ResolvedMounts[PlacedIndex].MountId == Mount.MountId;
    OutResult.Phase = Phase;
    OutResult.TerminationReason = TerminationReason;
    OutResult.StepIndex = StepIndex;
    OutResult.ActiveTargetCount = SpawnedTargets.Num();
    OutResult.DetectedTargetCount = LastDetectedTargetCount;
    OutResult.CurrentlyDetectedTargetCount = CurrentlyDetectedTargetCount;
    OutResult.TrackedTargetCount = TrackedTargetCount;
    OutResult.AllTargetsTrackedSteps = AllTargetsTrackedSteps;
    OutResult.StepRewards = LastStepRewards;
    OutResult.InvalidActionCount = InvalidActionCount;
    OutResult.LastBlueAction = LastBlueAction;
    OutResult.LastRedDeploymentAction = LastRedDeploymentAction;
    OutResult.LastRedMovementActions = LastRedMovementActions;
    OutResult.MinimumZoneDistanceMeters = ComputeMinimumZoneDistanceMeters();
    OutResult.BlueReward = AccumulatedBlueReward;
    OutResult.RedReward = AccumulatedRedReward;
    OutResult.bTerminal = Phase == ETRIADRLPhase::Terminal;
    for (int32 Index = 0; Index < ResolvedMounts.Num(); ++Index)
        OutResult.BlueActionMask.Add(CanPlaceCandidate(Index));
    OutResult.BlueActionMask.Add(Phase == ETRIADRLPhase::BluePlacement); // final action = commit

    constexpr double EarthRadiusMeters = 6378137.0;
    const double ZoneLatitudeRadians = FMath::DegreesToRadians(TrainingConfig.ProtectedZone.LatitudeDegrees);
    const double CosZoneLatitude = FMath::Max(FMath::Abs(FMath::Cos(ZoneLatitudeRadians)), 0.000001);
    const double SensorPositionScale = FMath::Max(TrainingConfig.BluePlacementRadiusMeters, 1.0);
    OutResult.SensorObservations.Reserve(SpawnedSensors.Num());
    for (const ATRIADSensorNodeActor* Sensor : SpawnedSensors)
    {
        if (!Sensor) continue;
        const FTRIADGeodeticSensorNode& Definition = Sensor->GetNodeDefinition();
        OutResult.SensorObservations.Add(FVector2D(
            FMath::DegreesToRadians(Definition.LongitudeDegrees - TrainingConfig.ProtectedZone.LongitudeDegrees) *
                EarthRadiusMeters * CosZoneLatitude / SensorPositionScale,
            FMath::DegreesToRadians(Definition.LatitudeDegrees - TrainingConfig.ProtectedZone.LatitudeDegrees) *
                EarthRadiusMeters / SensorPositionScale));
    }
    const double PositionScale = FMath::Max(TrainingConfig.MaximumDistanceFromZoneMeters, 1.0);
    const double VelocityScale = FMath::Max(TrainingConfig.MaximumTargetSpeedMetersPerSecond, 0.1);
    OutResult.TargetObservations.Reserve(TargetControllers.Num());
    for (int32 Index = 0; Index < TargetControllers.Num(); ++Index)
    {
        const UTRIADSwarmControllerComponent* Controller = TargetControllers[Index];
        if (!Controller) continue;
        FTRIADRLTargetObservation Observation;
        Observation.NormalizedZoneRelativeEnu = FVector(
            FMath::DegreesToRadians(Controller->GetLongitudeDegrees() - TrainingConfig.ProtectedZone.LongitudeDegrees) *
                EarthRadiusMeters * CosZoneLatitude / PositionScale,
            FMath::DegreesToRadians(Controller->GetLatitudeDegrees() - TrainingConfig.ProtectedZone.LatitudeDegrees) *
                EarthRadiusMeters / PositionScale,
            (Controller->GetHeightMeters() - TrainingConfig.ProtectedZone.HeightMeters) / PositionScale);
        Observation.NormalizedZoneRelativeEnu.X = FMath::Clamp(Observation.NormalizedZoneRelativeEnu.X, -1.5, 1.5);
        Observation.NormalizedZoneRelativeEnu.Y = FMath::Clamp(Observation.NormalizedZoneRelativeEnu.Y, -1.5, 1.5);
        Observation.NormalizedZoneRelativeEnu.Z = FMath::Clamp(Observation.NormalizedZoneRelativeEnu.Z, -1.5, 1.5);
        Observation.NormalizedVelocityEnu = Controller->GetVelocityEnuMetersPerSecond() / VelocityScale;
        const auto& Track = TargetTracks[Index];
        Observation.bConfirmedDetected = Track.bEverDetected;
        Observation.bCurrentlyDetected = Track.bCurrentlyDetected;
        Observation.bTracked = Track.bTracked;
        Observation.ConsecutiveDetectionSteps = Track.ConsecutiveDetectionSteps;
        Observation.FirstDetectionStep = Track.FirstDetectionStep;
        Observation.LastDetectionStep = Track.LastDetectionStep;
        Observation.TrackedStepCount = Track.TrackedStepCount;
        OutResult.TargetObservations.Add(Observation);
    }
}

ACesiumGeoreference* ATRIADAdversarialTrainingManager::FindGeoreference() const
{
    if (!GetWorld()) return nullptr;
    for (TActorIterator<ACesiumGeoreference> It(GetWorld()); It; ++It)
    {
        return *It;
    }
    return nullptr;
}

bool ATRIADAdversarialTrainingManager::GetEpisodeSnapshot(FTRIADRLStepResult& OutResult, FString& OutError) const
{
    BuildStepResult(OutResult);
    OutError.Reset();
    return true;
}

bool ATRIADAdversarialTrainingManager::SetEvaluationLabels(const FString& BlueCheckpoint, const FString& RedCheckpoint,
    FTRIADRLStepResult& OutResult, FString& OutError)
{
    if (!bEvaluationOnly || BlueCheckpoint.IsEmpty() || RedCheckpoint.IsEmpty() ||
        BlueCheckpoint.Len() > 120 || RedCheckpoint.Len() > 120 ||
        BlueCheckpoint.Contains(TEXT("\n")) || RedCheckpoint.Contains(TEXT("\n")))
    {
        OutError = TEXT("Checkpoint labels require evaluation mode and nonempty single-line labels up to 120 characters.");
        return false;
    }
    BluePolicyLabel = BlueCheckpoint;
    RedPolicyLabel = RedCheckpoint;
    BuildStepResult(OutResult);
    OutError.Reset();
    return true;
}

void ATRIADAdversarialTrainingManager::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);
    DrawEvaluationOverlay(); // drawing never advances a simulation step or reward
}

void ATRIADAdversarialTrainingManager::DrawEvaluationOverlay()
{
    if (!Georeference || !ObjectiveActor || !GetWorld()) return;
    DrawDebugCylinder(GetWorld(), ObjectiveActor->GetActorLocation(),
        ObjectiveActor->GetActorLocation() + FVector(0, 0, 5000), TrainingConfig.ProtectedZone.RadiusMeters * 100,
        48, FColor(255, 90, 70), false, -1.0f, 0, 2.0f);
    for (int32 Index = 0; Index < SpawnedSensors.Num(); ++Index)
    {
        const int32 OptionIndex = PlacedCatalogueIndices[Index];
        const auto& Option = TrainingConfig.SensorCandidates[OptionIndex];
        const FVector Position = SpawnedSensors[Index]->GetActorLocation();
        const FRotator Rotation = Georeference->TransformEastSouthUpRotatorToUnreal(
            FRotator(Option.PitchDegrees, Option.YawDegrees - 90.0, 0), Position);
        const FVector Forward = Rotation.Vector();
        const FColor Color = SensorDetectionCounts[Index] > 0 ? FColor::Green : FColor::Cyan;
        DrawDebugSphere(GetWorld(), Position, 70.0f, 12, Color, false, -1.0f, 0, 2.0f);
        DrawDebugDirectionalArrow(GetWorld(), Position, Position + Forward * 700, 150, Color, false, -1.0f, 0, 3.0f);
        const double Range = ResolvedMounts[OptionIndex].RangeMeters * 100.0;
        if (Option.HorizontalFovDegrees >= 359.0)
            DrawDebugSphere(GetWorld(), Position, Range, 32, Color, false, -1.0f, 0, 0.5f);
        else
            DrawDebugCone(GetWorld(), Position, Forward, Range,
                FMath::DegreesToRadians(FMath::Min(Option.HorizontalFovDegrees * 0.5, 89.9)),
                FMath::DegreesToRadians(FMath::Min(Option.VerticalFovDegrees * 0.5, 89.9)), 32, Color, false, -1.0f, 0, 1.0f);
        DrawDebugString(GetWorld(), Position + FVector(0, 0, 200),
            FString::Printf(TEXT("BLUE %s | yaw %.0f pitch %.0f | FOV %.0fx%.0f | %.0fm"),
                *Option.CandidateId, Option.YawDegrees, Option.PitchDegrees,
                Option.HorizontalFovDegrees, Option.VerticalFovDegrees, Range * 0.01), nullptr, Color, 0, true, 0.8f);
    }
    for (int32 Index = 0; Index < SpawnedTargets.Num(); ++Index)
    {
        const auto& Track = TargetTracks[Index];
        const FColor Color = Track.bTracked ? FColor::Green : (Track.bCurrentlyDetected ? FColor::Yellow : FColor::Red);
        const auto& Path = TargetPaths[Index];
        // Full paths are retained; drawing is decimated for very long episodes.
        const int32 Stride = FMath::Max(1, Path.Num() / 512);
        for (int32 Point = Stride; Point < Path.Num(); Point += Stride)
            DrawDebugLine(GetWorld(), Path[Point - Stride], Path[Point], Color, false, -1.0f, 0, 2.0f);
        if (Path.Num() > 1)
            DrawDebugLine(GetWorld(), Path[FMath::Max(0, ((Path.Num()-1) / Stride) * Stride)], Path.Last(), Color, false, -1.0f, 0, 2.0f);
        const FVector Position = SpawnedTargets[Index]->GetActorLocation();
        DrawDebugSphere(GetWorld(), Position, 110, 12, Color, false, -1.0f, 0, 2.0f);
        const FString Action = LastRedMovementActions.IsValidIndex(Index) ? LastRedMovementActions[Index].ToCompactString() : TEXT("waiting");
        DrawDebugString(GetWorld(), Position + FVector(0, 0, 250), FString::Printf(TEXT("RED %d %s | hits %d/%d | action %s"),
            Index, Track.bTracked ? TEXT("TRACKED") : (Track.bCurrentlyDetected ? TEXT("DETECTED") : TEXT("UNDETECTED")),
            Track.ConsecutiveDetectionSteps, TrainingConfig.TrackConfirmationSteps, *Action), nullptr, Color, 0, true, 1.0f);
    }
    if (GEngine)
    {
        const uint64 Key = static_cast<uint64>(GetUniqueID()) << 8;
        const FString PhaseLabel = StaticEnum<ETRIADRLPhase>()->GetNameStringByValue(static_cast<int64>(Phase));
        const FString ResultLabel = StaticEnum<ETRIADRLTerminationReason>()->GetNameStringByValue(static_cast<int64>(TerminationReason));
        GEngine->AddOnScreenDebugMessage(Key, 0.2f, FColor::White, FString::Printf(
            TEXT("TRIAD RL %s | %s | seed %d | step %d/%d | result %s"),
            bEvaluationOnly ? TEXT("EVALUATION - NO TRAINING") : TEXT("TRAINING DEBUG"), *PhaseLabel,
            EpisodeSeed, StepIndex, TrainingConfig.EpisodeHorizonSteps, *ResultLabel));
        GEngine->AddOnScreenDebugMessage(Key+1, 0.2f, FColor::Cyan, FString::Printf(
            TEXT("BLUE %s | action %s | budget %.2f/%.2f | return %.3f (step %.3f)"),
            *BluePolicyLabel, *LastBlueAction, TrainingConfig.SensorBudgetUnits-SpentSensorBudgetUnits,
            TrainingConfig.SensorBudgetUnits, AccumulatedBlueReward, LastStepRewards.BlueTotal()));
        GEngine->AddOnScreenDebugMessage(Key+2, 0.2f, FColor::Orange, FString::Printf(
            TEXT("RED %s | deploy %s | return %.3f (step %.3f)"),
            *RedPolicyLabel, *LastRedDeploymentAction, AccumulatedRedReward, LastStepRewards.RedTotal()));
        GEngine->AddOnScreenDebugMessage(Key+3, 0.2f, FColor::Green, FString::Printf(
            TEXT("Detected now %d | tracked %d/%d | defence hold %d/%d | invalid %d | cyan sensor / red unseen / yellow seen / green tracked"),
            CurrentlyDetectedTargetCount, TrackedTargetCount, SpawnedTargets.Num(), AllTargetsTrackedSteps,
            TrainingConfig.DefenceTrackHoldSteps, InvalidActionCount));
    }
}

void ATRIADAdversarialTrainingManager::LogTerminalEvaluation(const FTRIADRLStepResult& Result) const
{
    auto Record = FJsonObjectConverter::UStructToJsonObject(Result);
    const double Targets = FMath::Max(TargetTracks.Num(), 1);
    double LatencySum = 0.0, ContinuitySum = 0.0;
    int32 Detected = 0, TrackedSteps = 0;
    for (const auto& Track : TargetTracks)
    {
        TrackedSteps += Track.TrackedStepCount;
        if (!Track.bEverDetected) continue;
        ++Detected;
        LatencySum += Track.FirstDetectionStep * TrainingConfig.FixedStepSeconds;
        ContinuitySum += static_cast<double>(Track.TrackedStepCount) / FMath::Max(StepIndex - Track.FirstDetectionStep + 1, 1);
    }
    const bool bBreach = TerminationReason == ETRIADRLTerminationReason::ProtectedZoneReached;
    Record->SetStringField(TEXT("blueCheckpoint"), BluePolicyLabel);
    Record->SetStringField(TEXT("redCheckpoint"), RedPolicyLabel);
    Record->SetNumberField(TEXT("breachRate"), bBreach ? 1.0 : 0.0);
    Record->SetNumberField(TEXT("blueWinRate"), bBreach ? 0.0 : 1.0);
    Record->SetNumberField(TEXT("redWinRate"), bBreach ? 1.0 : 0.0);
    Record->SetNumberField(TEXT("detectionRate"), Detected / Targets);
    Record->SetNumberField(TEXT("meanDetectionLatencySecondsDetectedTargets"), Detected ? LatencySum / Detected : -1.0);
    Record->SetNumberField(TEXT("trackContinuityAfterFirstDetection"), Detected ? ContinuitySum / Detected : 0.0);
    Record->SetNumberField(TEXT("trackingCoverage"), TrackedSteps / (Targets * FMath::Max(StepIndex, 1)));
    Record->SetStringField(TEXT("defenceSemantics"), TEXT("Sustained observed track or survived horizon; not physical interception"));
    FString Json;
    const auto Writer = TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Json);
    FJsonSerializer::Serialize(Record.ToSharedRef(), Writer);
    const FString Directory = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("TRIAD/RL/Evaluation"));
    IFileManager::Get().MakeDirectory(*Directory, true);
    if (!FFileHelper::SaveStringToFile(Json + TEXT("\n"), *FPaths::Combine(Directory, TEXT("episodes.jsonl")),
        FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM, &IFileManager::Get(), FILEWRITE_Append))
        UE_LOG(LogTemp, Error, TEXT("TRIAD RL could not append evaluation episode metrics."));
}
