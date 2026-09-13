#include "TRIADRLTrainingModel.h"

bool FTRIADRLTrainingModel::ValidateConfig(const FTRIADRLTrainingConfig& Config, FString& OutError)
{
    if (Config.SchemaVersion != TEXT("triad.rl_training.v4"))
    {
        OutError = TEXT("RL definition requires schemaVersion triad.rl_training.v4 and a dynamic sensor catalogue.");
        return false;
    }
    if (!FMath::IsFinite(Config.FixedStepSeconds) || Config.FixedStepSeconds < 0.02 || Config.FixedStepSeconds > 2.0 ||
        Config.EpisodeHorizonSteps < 1 || Config.EpisodeHorizonSteps > 100000)
    {
        OutError = TEXT("RL fixed step or episode horizon is outside the bounded contract.");
        return false;
    }
    if (!FMath::IsFinite(Config.ProtectedZone.LongitudeDegrees) ||
        !FMath::IsFinite(Config.ProtectedZone.LatitudeDegrees) ||
        FMath::Abs(Config.ProtectedZone.LongitudeDegrees) > 180.0 ||
        FMath::Abs(Config.ProtectedZone.LatitudeDegrees) >= 89.0 ||
        !FMath::IsFinite(Config.ProtectedZone.HeightMeters) ||
        !FMath::IsFinite(Config.ProtectedZone.RadiusMeters) || Config.ProtectedZone.RadiusMeters < 10.0)
    {
        OutError = TEXT("RL protected-zone definition is invalid.");
        return false;
    }
    if (Config.MaximumSensorSites < 1 || Config.MaximumSensorSites > 32 ||
        !FMath::IsFinite(Config.SensorBudgetUnits) || Config.SensorBudgetUnits < 0.0 ||
        !FMath::IsFinite(Config.BluePlacementRadiusMeters) || Config.BluePlacementRadiusMeters < 50.0 ||
        !FMath::IsFinite(Config.BlueMinimumObjectiveStandoffMeters) ||
        Config.BlueMinimumObjectiveStandoffMeters < 0.0 ||
        Config.BlueMinimumObjectiveStandoffMeters >= Config.BluePlacementRadiusMeters ||
        !FMath::IsFinite(Config.MinimumSensorSeparationMeters) || Config.MinimumSensorSeparationMeters < 0.0 ||
        !FMath::IsFinite(Config.RedMinimumSpawnRadiusMeters) ||
        !FMath::IsFinite(Config.RedMaximumSpawnRadiusMeters) ||
        Config.RedMinimumSpawnRadiusMeters <= Config.BluePlacementRadiusMeters ||
        Config.RedMaximumSpawnRadiusMeters < Config.RedMinimumSpawnRadiusMeters ||
        !FMath::IsFinite(Config.RedMinimumAltitudeMeters) ||
        !FMath::IsFinite(Config.RedMaximumAltitudeMeters) ||
        Config.RedMinimumAltitudeMeters < 5.0 || Config.RedMaximumAltitudeMeters < Config.RedMinimumAltitudeMeters ||
        Config.RedMinimumSwarmSize < 1 || Config.RedMaximumSwarmSize > 64 ||
        Config.RedMaximumSwarmSize < Config.RedMinimumSwarmSize ||
        !FMath::IsFinite(Config.RedMinimumFormationSpacingMeters) ||
        !FMath::IsFinite(Config.RedMaximumFormationSpacingMeters) ||
        Config.RedMinimumFormationSpacingMeters < 1.0 ||
        Config.RedMaximumFormationSpacingMeters < Config.RedMinimumFormationSpacingMeters ||
        !FMath::IsFinite(Config.MaximumTargetSpeedMetersPerSecond) ||
        Config.MaximumTargetSpeedMetersPerSecond < 0.1 || Config.MaximumTargetSpeedMetersPerSecond > 50.0 ||
        !FMath::IsFinite(Config.MaximumDistanceFromZoneMeters) || Config.MaximumDistanceFromZoneMeters < 100.0)
    {
        OutError = TEXT("RL continuous placement, deployment, budget, speed, or AOI bound is invalid.");
        return false;
    }
    if (Config.ScenarioId.IsEmpty() || Config.SensorCandidates.IsEmpty() || Config.SensorCandidates.Num() > 256 ||
        Config.ConfirmationNodeCount < 1 || Config.ConfirmationNodeCount > Config.MaximumSensorSites ||
        !FMath::IsFinite(Config.RedMinimumBearingDegrees) || !FMath::IsFinite(Config.RedMaximumBearingDegrees) ||
        Config.RedMinimumBearingDegrees < -180.0 || Config.RedMaximumBearingDegrees > 180.0 ||
        Config.RedMinimumBearingDegrees > Config.RedMaximumBearingDegrees)
    {
        OutError = TEXT("RL requires a named scenario, 1..256 catalogue options, feasible confirmation count and bounded bearings.");
        return false;
    }
    TSet<FString> CandidateIds;
    TMap<FString, FVector> MountPositions;
    TMap<FString, double> FixedMountMinimumCosts;
    double DynamicMinimumCost = TNumericLimits<double>::Max();
    for (const FTRIADRLSensorCandidate& Option : Config.SensorCandidates)
    {
        const double Scalars[] = {Option.OffsetEnuMeters.X, Option.OffsetEnuMeters.Y, Option.OffsetEnuMeters.Z,
            Option.YawDegrees, Option.PitchDegrees, Option.HorizontalFovDegrees, Option.VerticalFovDegrees,
            Option.CostUnits, Option.Sensor.DetectionRangeMeters, Option.Sensor.SearchRadarRangeMeters,
            Option.Sensor.EOPTZConfirmationRangeMeters, Option.Sensor.ThermalPTZConfirmationRangeMeters};
        bool bFinite = true;
        for (double Value : Scalars) bFinite &= FMath::IsFinite(Value);
        const double Radius = FVector2D(Option.OffsetEnuMeters.X, Option.OffsetEnuMeters.Y).Length();
        const FVector* ExistingMount = MountPositions.Find(Option.MountId);
        if (!bFinite || Option.CandidateId.IsEmpty() || Option.SensorProfileId.IsEmpty() || Option.MountId.IsEmpty() ||
            CandidateIds.Contains(Option.CandidateId) || Option.CostUnits <= 0.0 ||
            Option.HorizontalFovDegrees <= 0.0 || Option.HorizontalFovDegrees > 360.0 ||
            Option.VerticalFovDegrees <= 0.0 || Option.VerticalFovDegrees > 180.0 ||
            FMath::Abs(Option.PitchDegrees) > 90.0 ||
            (!Option.bAllowDynamicPosition && (Radius < Config.BlueMinimumObjectiveStandoffMeters ||
                Radius > Config.BluePlacementRadiusMeters)) ||
            (!Option.bAllowDynamicPosition && ExistingMount && !ExistingMount->Equals(Option.OffsetEnuMeters, 0.01)) ||
            GetSensorModalityMask(Option.Sensor) == 0)
        {
            OutError = TEXT("RL catalogue contains a duplicate, sensorless, inconsistent, non-finite or out-of-domain option.");
            return false;
        }
        if (Option.Sensor.bEnableSearchRadar && Option.Sensor.SearchRadarRangeMeters <= 0.0)
        {
            OutError = TEXT("An enabled search-radar profile requires a positive range.");
            return false;
        }
        if (Option.Sensor.bEnableEOPTZ && Option.Sensor.EOPTZConfirmationRangeMeters <= 0.0)
        {
            OutError = TEXT("An enabled EO profile requires a positive confirmation range.");
            return false;
        }
        if (Option.Sensor.bEnableThermalPTZ && Option.Sensor.ThermalPTZConfirmationRangeMeters <= 0.0)
        {
            OutError = TEXT("An enabled thermal profile requires a positive confirmation range.");
            return false;
        }
        for (double FrequencyGHz : Option.Sensor.SupportedFrequenciesGHz)
        {
            if (!FMath::IsFinite(FrequencyGHz) || FrequencyGHz <= 0.0)
            {
                OutError = TEXT("Passive-RF profile frequencies must be finite and positive.");
                return false;
            }
        }
        CandidateIds.Add(Option.CandidateId);
        if (!Option.bAllowDynamicPosition) MountPositions.Add(Option.MountId, Option.OffsetEnuMeters);
        if (Option.bApproved && Option.Sensor.bEnabled)
        {
            if (Option.bAllowDynamicPosition)
            {
                DynamicMinimumCost = FMath::Min(DynamicMinimumCost, Option.CostUnits);
            }
            else
            {
                double& Cost = FixedMountMinimumCosts.FindOrAdd(Option.MountId, Option.CostUnits);
                Cost = FMath::Min(Cost, Option.CostUnits);
            }
        }
    }
    TArray<double> FeasiblePlacementCosts;
    FixedMountMinimumCosts.GenerateValueArray(FeasiblePlacementCosts);
    if (DynamicMinimumCost < TNumericLimits<double>::Max())
    {
        for (int32 Index = 0; Index < Config.MaximumSensorSites; ++Index)
            FeasiblePlacementCosts.Add(DynamicMinimumCost);
    }
    FeasiblePlacementCosts.Sort();
    if (FeasiblePlacementCosts.Num() < Config.ConfirmationNodeCount)
    {
        OutError = TEXT("RL confirmation count exceeds the available enabled dynamic profiles or distinct fixed mounts.");
        return false;
    }
    double MinimumConfirmationCost = 0.0;
    for (int32 Index = 0; Index < Config.ConfirmationNodeCount; ++Index)
        MinimumConfirmationCost += FeasiblePlacementCosts[Index];
    if (MinimumConfirmationCost > Config.SensorBudgetUnits + UE_DOUBLE_SMALL_NUMBER)
    {
        OutError = TEXT("RL sensor budget cannot fund the configured confirmation count.");
        return false;
    }
    if (Config.TrackConfirmationSteps < 1 || Config.DefenceTrackHoldSteps < 1 ||
        Config.TrackConfirmationSteps > Config.EpisodeHorizonSteps || Config.DefenceTrackHoldSteps > Config.EpisodeHorizonSteps ||
        Config.TrackConfirmationSteps + Config.DefenceTrackHoldSteps - 1 > Config.EpisodeHorizonSteps ||
        Config.Difficulty < 0 || Config.RedMovementAxisMask.IsNearlyZero())
    {
        OutError = TEXT("RL tracking horizon, difficulty or movement axes are invalid.");
        return false;
    }
    for (int32 Axis = 0; Axis < 3; ++Axis)
        if (Config.RedMovementAxisMask[Axis] != 0.0 && Config.RedMovementAxisMask[Axis] != 1.0)
        {
            OutError = TEXT("RL movement-axis mask must contain only zero or one.");
            return false;
        }
    const auto& W = Config.Rewards;
    const double AllWeights[] = {W.BlueDetection, W.BlueEarlyDetection, W.BlueTracking, W.BlueDefenceWin,
        W.BlueZoneMiss, W.BlueSiteCost, W.RedProgress, W.RedDetected, W.RedTracking, W.RedTime,
        W.RedDefenceLoss, W.RedZoneReached, W.RedConstraintViolation};
    for (double Value : AllWeights)
        if (!FMath::IsFinite(Value)) { OutError = TEXT("RL rewards must be finite."); return false; }
    // Shaping is bounded over the WHOLE episode, independently of swarm size,
    // site count or horizon. Terminal success/loss must dominate that bound.
    const double BlueBound = W.BlueDetection + W.BlueEarlyDetection + W.BlueTracking - W.BlueSiteCost;
    const double RedBound = W.RedProgress - W.RedDetected - W.RedTracking - W.RedTime;
    if (W.BlueDetection < 0 || W.BlueEarlyDetection < 0 || W.BlueTracking < 0 || W.BlueSiteCost > 0 ||
        W.RedProgress < 0 || W.RedDetected > 0 || W.RedTracking > 0 || W.RedTime > 0 ||
        W.BlueDefenceWin <= BlueBound || -W.BlueZoneMiss <= BlueBound ||
        W.RedZoneReached <= RedBound || -W.RedDefenceLoss <= RedBound || -W.RedConstraintViolation <= RedBound)
    {
        OutError = TEXT("RL terminal rewards must dominate all bounded shaping and use the intended signs.");
        return false;
    }
    OutError.Reset();
    return true;
}

int32 FTRIADRLTrainingModel::GetSensorModalityMask(const FTRIADGeodeticSensorNode& Sensor)
{
    int32 Mask = 0;
    if (Sensor.DetectionRangeMeters > 0.0 && !Sensor.SupportedFrequenciesGHz.IsEmpty())
        Mask |= static_cast<int32>(ETRIADRLSensorModality::PassiveRF);
    if (Sensor.bEnableSearchRadar)
        Mask |= static_cast<int32>(ETRIADRLSensorModality::SearchRadar);
    if (Sensor.bEnableEOPTZ)
        Mask |= static_cast<int32>(ETRIADRLSensorModality::ElectroOptical);
    if (Sensor.bEnableThermalPTZ)
        Mask |= static_cast<int32>(ETRIADRLSensorModality::Thermal);
    return Mask;
}

double FTRIADRLTrainingModel::GreatCircleDistanceMeters(
    double LongitudeA, double LatitudeA, double LongitudeB, double LatitudeB)
{
    constexpr double EarthRadiusMeters = 6378137.0;
    const double LatitudeARadians = FMath::DegreesToRadians(LatitudeA);
    const double LatitudeBRadians = FMath::DegreesToRadians(LatitudeB);
    const double DeltaLatitude = LatitudeBRadians - LatitudeARadians;
    const double DeltaLongitude = FMath::DegreesToRadians(LongitudeB - LongitudeA);
    const double Haversine = FMath::Square(FMath::Sin(DeltaLatitude * 0.5)) +
        FMath::Cos(LatitudeARadians) * FMath::Cos(LatitudeBRadians) *
        FMath::Square(FMath::Sin(DeltaLongitude * 0.5));
    return EarthRadiusMeters * 2.0 * FMath::Asin(FMath::Sqrt(FMath::Clamp(Haversine, 0.0, 1.0)));
}

FVector FTRIADRLTrainingModel::ClampNormalizedAction(const FVector& Action)
{
    if (!FMath::IsFinite(Action.X) || !FMath::IsFinite(Action.Y) || !FMath::IsFinite(Action.Z))
    {
        return FVector::ZeroVector;
    }
    return FVector(
        FMath::Clamp(Action.X, -1.0, 1.0),
        FMath::Clamp(Action.Y, -1.0, 1.0),
        FMath::Clamp(Action.Z, -1.0, 1.0));
}

bool FTRIADRLTrainingModel::AdvanceTrack(FTRIADRLTrackState& Track, bool bDetected, int32 StepIndex, int32 ConfirmationSteps)
{
    if (StepIndex <= Track.LastUpdateStep) return false;
    Track.LastUpdateStep = StepIndex;
    const bool bFirstDetection = bDetected && !Track.bEverDetected;
    Track.bCurrentlyDetected = bDetected;
    Track.ConsecutiveDetectionSteps = bDetected ? Track.ConsecutiveDetectionSteps + 1 : 0;
    Track.bTracked = bDetected && Track.ConsecutiveDetectionSteps >= ConfirmationSteps;
    if (bDetected)
    {
        Track.bEverDetected = true;
        Track.LastDetectionStep = StepIndex;
        ++Track.DetectedStepCount;
        if (bFirstDetection) Track.FirstDetectionStep = StepIndex;
    }
    if (Track.bTracked) ++Track.TrackedStepCount;
    return bFirstDetection;
}

void FTRIADRLTrainingModel::ComputeStepRewards(
    const FTRIADRLTrainingConfig& Config,
    double PreviousZoneDistanceMeters,
    double CurrentZoneDistanceMeters,
    double NewlyDetectedFraction,
    double EarlyDetectionQuality,
    double TrackedFraction,
    ETRIADRLTerminationReason TerminalReason,
    FTRIADRLRewardBreakdown& OutRewards)
{
    OutRewards = FTRIADRLRewardBreakdown();
    const auto& Weights = Config.Rewards;
    const double Scale = FMath::Max(Config.MaximumDistanceFromZoneMeters, 1.0);
    // Fixed potential difference telescopes. Moving out and back cannot farm
    // reward as it could when dividing by the previous step's shrinking distance.
    const double ProgressFraction = FMath::Clamp(PreviousZoneDistanceMeters / Scale, 0.0, 1.0) -
        FMath::Clamp(CurrentZoneDistanceMeters / Scale, 0.0, 1.0);
    const double NewFraction = FMath::Clamp(NewlyDetectedFraction, 0.0, 1.0);
    const double TrackStep = FMath::Clamp(TrackedFraction, 0.0, 1.0) / FMath::Max(Config.EpisodeHorizonSteps, 1);
    OutRewards.BlueDetection = NewFraction * Weights.BlueDetection;
    OutRewards.BlueEarlyDetection = NewFraction * FMath::Clamp(EarlyDetectionQuality, 0.0, 1.0) * Weights.BlueEarlyDetection;
    OutRewards.BlueTracking = TrackStep * Weights.BlueTracking;
    OutRewards.RedProgress = ProgressFraction * Weights.RedProgress;
    OutRewards.RedDetection = NewFraction * Weights.RedDetected;
    OutRewards.RedTracking = TrackStep * Weights.RedTracking;
    OutRewards.RedTime = Weights.RedTime / FMath::Max(Config.EpisodeHorizonSteps, 1);
    if (TerminalReason == ETRIADRLTerminationReason::ProtectedZoneReached)
    {
        OutRewards.BlueTerminal = Weights.BlueZoneMiss;
        OutRewards.RedTerminal = Weights.RedZoneReached;
    }
    else if (TerminalReason != ETRIADRLTerminationReason::None)
    {
        OutRewards.BlueTerminal = Weights.BlueDefenceWin;
        OutRewards.RedTerminal = TerminalReason == ETRIADRLTerminationReason::ConstraintViolation
            ? Weights.RedConstraintViolation : Weights.RedDefenceLoss;
    }
}
