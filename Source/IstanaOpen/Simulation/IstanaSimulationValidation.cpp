#include "Simulation/IstanaSimulationValidation.h"

namespace
{
    bool FiniteVector(const FVector& V)
    {
        return FMath::IsFinite(V.X) && FMath::IsFinite(V.Y) && FMath::IsFinite(V.Z);
    }

    struct FChecks
    {
        TArray<FIstanaValidationIssue>& Issues;
        void Require(bool bValid, const FString& Path, const TCHAR* Message)
        {
            if (!bValid)
            {
                FIstanaValidationIssue& Issue = Issues.AddDefaulted_GetRef();
                Issue.FieldPath = Path;
                Issue.Message = Message;
            }
        }
        void Nonnegative(double Value, const FString& Path)
        {
            Require(FMath::IsFinite(Value) && Value >= 0, Path, TEXT("Must be finite and nonnegative."));
        }
        void Probability(double Value, const FString& Path)
        {
            Require(FMath::IsFinite(Value) && Value >= 0 && Value <= 1, Path, TEXT("Must be finite and between 0 and 1."));
        }
    };

    void CheckSensor(const FIstanaSensorConfig& S, const FString& Prefix, FChecks& Check)
    {
        Check.Require(S.SensorId >= 0, Prefix + TEXT("SensorId"), TEXT("Must be a nonnegative episode-local ID."));
        Check.Require(S.SensorType == EIstanaSensorType::AbstractCoverage, Prefix + TEXT("SensorType"), TEXT("Unsupported sensor model."));
        const FQuat Rotation = S.Transform.GetRotation();
        Check.Require(FiniteVector(S.Transform.GetLocation()) && FiniteVector(S.Transform.GetScale3D())
            && FMath::IsFinite(Rotation.X) && FMath::IsFinite(Rotation.Y)
            && FMath::IsFinite(Rotation.Z) && FMath::IsFinite(Rotation.W)
            && Rotation.IsNormalized(), Prefix + TEXT("Transform"), TEXT("Requires finite values and a normalized rotation."));
        Check.Require(S.Transform.GetScale3D().Equals(FVector::OneVector), Prefix + TEXT("Transform.Scale3D"), TEXT("Sensor transforms must have unit scale; configure coverage in Model."));
        Check.Require(S.SampleEveryNSteps > 0, Prefix + TEXT("SampleEveryNSteps"), TEXT("Must be at least one step."));
        Check.Nonnegative(S.Model.CoverageRadiusCm, Prefix + TEXT("Model.CoverageRadiusCm"));
        Check.Nonnegative(S.Model.PositionNoiseStdDevCm, Prefix + TEXT("Model.PositionNoiseStdDevCm"));
        Check.Probability(S.Model.DetectionProbabilityPerSample, Prefix + TEXT("Model.DetectionProbabilityPerSample"));
        Check.Probability(S.Model.FalsePositiveProbabilityPerSample, Prefix + TEXT("Model.FalsePositiveProbabilityPerSample"));
        Check.Require(S.Model.LatencySteps >= 0, Prefix + TEXT("Model.LatencySteps"), TEXT("Cannot be negative."));
    }
}

bool UIstanaSimulationValidation::ValidateScenario(const FIstanaScenarioConfig& Config, TArray<FIstanaValidationIssue>& Issues)
{
    Issues.Reset();
    FChecks Check{Issues};
    Check.Require(Config.SchemaVersion == 1, TEXT("SchemaVersion"), TEXT("Only schema version 1 is supported."));
    Check.Require(!Config.ScenarioId.IsNone(), TEXT("ScenarioId"), TEXT("A stable scenario identifier is required."));
    Check.Require(!Config.Map.IsNull() && Config.Map.ToSoftObjectPath().IsValid(), TEXT("Map"), TEXT("Assign a fictional simulation map. Asset existence is checked by the loader."));
    Check.Require(FMath::IsFinite(Config.FixedStepSeconds) && Config.FixedStepSeconds > 0, TEXT("FixedStepSeconds"), TEXT("Must be finite and positive."));
    Check.Require(FMath::IsFinite(Config.DurationSeconds) && Config.DurationSeconds > 0
        && Config.DurationSeconds >= Config.FixedStepSeconds, TEXT("DurationSeconds"), TEXT("Must be finite, positive, and at least one simulation step."));
    Check.Require(Config.SensorBudget >= 0, TEXT("SensorBudget"), TEXT("Cannot be negative."));
    Check.Require(Config.InitialSensors.Num() <= Config.SensorBudget, TEXT("InitialSensors"), TEXT("Initial sensor count exceeds the budget."));
    TSet<int32> SensorIds;
    for (int32 Index = 0; Index < Config.InitialSensors.Num(); ++Index)
    {
        const FIstanaSensorConfig& Sensor = Config.InitialSensors[Index];
        const FString Prefix = FString::Printf(TEXT("InitialSensors[%d]."), Index);
        CheckSensor(Sensor, Prefix, Check);
        Check.Require(!SensorIds.Contains(Sensor.SensorId), Prefix + TEXT("SensorId"), TEXT("Duplicate sensor ID."));
        SensorIds.Add(Sensor.SensorId);
    }
    TSet<int32> GroupIds;
    for (int32 Index = 0; Index < Config.Swarms.Num(); ++Index)
    {
        const FIstanaSwarmConfig& Swarm = Config.Swarms[Index];
        const FString Prefix = FString::Printf(TEXT("Swarms[%d]."), Index);
        Check.Require(Swarm.GroupId >= 0 && !GroupIds.Contains(Swarm.GroupId), Prefix + TEXT("GroupId"), TEXT("Requires a unique nonnegative group ID."));
        GroupIds.Add(Swarm.GroupId);
        Check.Require(Swarm.DroneCount > 0, Prefix + TEXT("DroneCount"), TEXT("Must be positive; omit the group to configure zero drones."));
        Check.Require(FiniteVector(Swarm.SpawnOriginCm), Prefix + TEXT("SpawnOriginCm"), TEXT("Must contain finite coordinates."));
        Check.Nonnegative(Swarm.SpawnRadiusCm, Prefix + TEXT("SpawnRadiusCm"));
        Check.Require(!Swarm.MovementPresetId.IsNone(), Prefix + TEXT("MovementPresetId"), TEXT("A movement preset identifier is required."));
    }
    Check.Require(FMath::IsFinite(Config.Conditions.TimeOfDayHours) && Config.Conditions.TimeOfDayHours >= 0
        && Config.Conditions.TimeOfDayHours < 24, TEXT("Conditions.TimeOfDayHours"), TEXT("Must be in [0, 24); midnight is 0."));
    Check.Probability(Config.Conditions.PrecipitationFraction, TEXT("Conditions.PrecipitationFraction"));
    Check.Probability(Config.Conditions.VisibilityFraction, TEXT("Conditions.VisibilityFraction"));
    Check.Require(FiniteVector(Config.Conditions.WindVelocityCmPerSecond), TEXT("Conditions.WindVelocityCmPerSecond"), TEXT("Must contain finite components."));
    return Issues.IsEmpty();
}

bool UIstanaSimulationValidation::ValidateSensor(const FIstanaSensorConfig& Sensor, TArray<FIstanaValidationIssue>& Issues)
{
    Issues.Reset();
    FChecks Check{Issues};
    CheckSensor(Sensor, TEXT(""), Check);
    return Issues.IsEmpty();
}

bool UIstanaSimulationValidation::ValidateDroneStates(const FIstanaScenarioConfig& Config, const TArray<FIstanaDroneState>& States, TArray<FIstanaValidationIssue>& Issues)
{
    Issues.Reset();
    FChecks Check{Issues};
    TSet<int32> GroupIds;
    for (const FIstanaSwarmConfig& Swarm : Config.Swarms) GroupIds.Add(Swarm.GroupId);
    TSet<int32> DroneIds;
    for (int32 Index = 0; Index < States.Num(); ++Index)
    {
        const FIstanaDroneState& State = States[Index];
        const FString Prefix = FString::Printf(TEXT("States[%d]."), Index);
        Check.Require(State.DroneId >= 0 && !DroneIds.Contains(State.DroneId), Prefix + TEXT("DroneId"), TEXT("Requires a unique nonnegative drone ID."));
        DroneIds.Add(State.DroneId);
        Check.Require(GroupIds.Contains(State.GroupId), Prefix + TEXT("GroupId"), TEXT("Group ID is not present in the scenario."));
        Check.Require(FiniteVector(State.PositionCm), Prefix + TEXT("PositionCm"), TEXT("Must contain finite coordinates."));
        Check.Require(FiniteVector(State.VelocityCmPerSecond), Prefix + TEXT("VelocityCmPerSecond"), TEXT("Must contain finite components."));
    }
    return Issues.IsEmpty();
}

bool UIstanaSimulationValidation::ValidateDetection(const FIstanaDetectionEvent& Event, const TArray<int32>& KnownSensorIds, TArray<FIstanaValidationIssue>& Issues)
{
    Issues.Reset();
    FChecks Check{Issues};
    Check.Require(Event.SensorId >= 0 && KnownSensorIds.Contains(Event.SensorId), TEXT("SensorId"), TEXT("Unknown sensor ID."));
    Check.Require(Event.SequenceNumber >= 0, TEXT("SequenceNumber"), TEXT("Cannot be negative."));
    Check.Require(Event.SampleStep >= 0, TEXT("SampleStep"), TEXT("Cannot be negative."));
    Check.Require(Event.DeliveryStep >= Event.SampleStep, TEXT("DeliveryStep"), TEXT("Cannot precede the sample step."));
    Check.Probability(Event.Confidence, TEXT("Confidence"));
    Check.Require(!Event.Observation.bHasPosition || FiniteVector(Event.Observation.PositionCm), TEXT("Observation.PositionCm"), TEXT("An available position must contain finite coordinates."));
    // V1 reports position detections only; other measurement types need an explicit schema extension.
    Check.Require(Event.Observation.bHasPosition, TEXT("Observation.bHasPosition"), TEXT("V1 detections require a reported position."));
    return Issues.IsEmpty();
}

bool UIstanaSimulationValidation::ValidateEpisodeResult(const FIstanaEpisodeResult& Result, TArray<FIstanaValidationIssue>& Issues)
{
    Issues.Reset();
    FChecks Check{Issues};
    Check.Require(Result.SchemaVersion == 1, TEXT("SchemaVersion"), TEXT("Only schema version 1 is supported."));
    Check.Require(Result.RunId.IsValid(), TEXT("RunId"), TEXT("A run ID is required."));
    Check.Require(!Result.ScenarioId.IsNone(), TEXT("ScenarioId"), TEXT("A scenario ID is required."));
    Check.Require(!Result.SimulationVersion.IsEmpty(), TEXT("SimulationVersion"), TEXT("Record the simulator implementation version."));
    Check.Require(!Result.SensorModelVersion.IsEmpty(), TEXT("SensorModelVersion"), TEXT("Record the sensor model implementation version."));
    Check.Require(Result.ExecutedSteps >= 0, TEXT("ExecutedSteps"), TEXT("Cannot be negative."));
    Check.Require(FMath::IsFinite(Result.FixedStepSeconds) && Result.FixedStepSeconds > 0, TEXT("FixedStepSeconds"), TEXT("Must be finite and positive."));
    Check.Require(Result.CompletionReason >= EIstanaCompletionReason::DurationReached
        && Result.CompletionReason <= EIstanaCompletionReason::Failed, TEXT("CompletionReason"), TEXT("A final result requires a supported completion reason."));
    const FIstanaEpisodeMetrics& M = Result.Metrics;
    Check.Require(M.SensorSampleCount >= 0 && M.EligibleObjectSampleCount >= 0 && M.TrueDetectionCount >= 0
        && M.MissedDetectionCount >= 0 && M.FalseObservationCount >= 0, TEXT("Metrics"), TEXT("Counts cannot be negative."));
    // Subtraction after bounds checks avoids signed overflow on malformed counts.
    Check.Require(M.TrueDetectionCount >= 0 && M.TrueDetectionCount <= M.EligibleObjectSampleCount
        && M.MissedDetectionCount == M.EligibleObjectSampleCount - M.TrueDetectionCount,
        TEXT("Metrics.EligibleObjectSampleCount"), TEXT("Must equal true detections plus misses."));
    Check.Require(M.FalseObservationCount <= M.SensorSampleCount, TEXT("Metrics.FalseObservationCount"), TEXT("V1 permits at most one false observation per sensor sample."));
    Check.Require(M.bHasFirstDetection == (M.TrueDetectionCount > 0), TEXT("Metrics.bHasFirstDetection"), TEXT("Must indicate whether any true detection occurred."));
    Check.Require(!M.bHasFirstDetection || (M.FirstDetectionStep >= 0 && M.FirstDetectionStep < Result.ExecutedSteps),
        TEXT("Metrics.FirstDetectionStep"), TEXT("Must reference an executed step."));
    return Issues.IsEmpty();
}

bool UIstanaSimulationValidation::ValidateActionBatch(const FIstanaPolicyActionBatch& Actions, const FIstanaPolicyContext& Context, int64 ExpectedDecisionStep, TArray<FIstanaValidationIssue>& Issues)
{
    Issues.Reset();
    FChecks Check{Issues};
    Check.Require(Context.RunId.IsValid() && Actions.RunId == Context.RunId, TEXT("RunId"), TEXT("Actions must belong to the active episode."));
    Check.Require(ExpectedDecisionStep >= 0 && Actions.DecisionStep == ExpectedDecisionStep, TEXT("DecisionStep"), TEXT("Actions must match the current decision step."));
    Check.Require(Context.Role >= EIstanaPolicyRole::Observer && Context.Role <= EIstanaPolicyRole::Red, TEXT("Context.Role"), TEXT("Unsupported policy role."));
    Check.Require(FMath::IsFinite(Context.FixedStepSeconds) && Context.FixedStepSeconds > 0, TEXT("Context.FixedStepSeconds"), TEXT("Must be finite and positive."));
    return Issues.IsEmpty();
}

FIstanaScenarioConfig UIstanaSimulationValidation::MakeExampleScenario()
{
    FIstanaScenarioConfig Config;
    Config.ScenarioId = TEXT("SyntheticContractExample");
    FIstanaSwarmConfig Swarm;
    Swarm.GroupId = 0;
    Swarm.DroneCount = 3;
    Config.Swarms.Add(Swarm);
    FIstanaSensorConfig Sensor;
    Sensor.SensorId = 0;
    Config.InitialSensors.Add(Sensor);
    return Config;
}
