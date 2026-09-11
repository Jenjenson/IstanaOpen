#include "Simulation/IstanaSimulationValidation.h"
#include "Simulation/IstanaSimulationPresets.h"
#include "Simulation/IstanaPolicyInterface.h"
#include "Misc/AutomationTest.h"
#include <limits>

#if WITH_DEV_AUTOMATION_TESTS
namespace
{
    FIstanaScenarioConfig ValidScenario()
    {
        FIstanaScenarioConfig Config = UIstanaSimulationValidation::MakeExampleScenario();
        // Structural validation must not load or require an installed test map.
        Config.Map = TSoftObjectPtr<UWorld>(FSoftObjectPath(TEXT("/Game/Tests/SyntheticMap.SyntheticMap")));
        return Config;
    }

    bool HasIssue(const TArray<FIstanaValidationIssue>& Issues, const TCHAR* Path)
    {
        return Issues.ContainsByPredicate([Path](const FIstanaValidationIssue& Issue)
        {
            return Issue.FieldPath == Path;
        });
    }
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIstanaConfigValidationTest, "Istana.Simulation.Contracts.Configuration",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FIstanaConfigValidationTest::RunTest(const FString& Parameters)
{
    TArray<FIstanaValidationIssue> Issues;
    FIstanaScenarioConfig Config = ValidScenario();
    TestTrue(TEXT("Valid synthetic scenario"), UIstanaSimulationValidation::ValidateScenario(Config, Issues));
    const FIstanaSensorConfig DuplicateSensor = Config.InitialSensors[0];
    const FIstanaSwarmConfig DuplicateSwarm = Config.Swarms[0];
    Config.InitialSensors.Add(DuplicateSensor);
    Config.Swarms.Add(DuplicateSwarm);
    Config.FixedStepSeconds = 0;
    Config.Conditions.VisibilityFraction = std::numeric_limits<double>::quiet_NaN();
    TestFalse(TEXT("Invalid configuration rejected"), UIstanaSimulationValidation::ValidateScenario(Config, Issues));
    TestTrue(TEXT("Duplicate sensor field identified"), HasIssue(Issues, TEXT("InitialSensors[1].SensorId")));
    TestTrue(TEXT("Duplicate group field identified"), HasIssue(Issues, TEXT("Swarms[1].GroupId")));
    TestTrue(TEXT("Budget violation identified"), HasIssue(Issues, TEXT("InitialSensors")));
    TestTrue(TEXT("Zero timestep identified"), HasIssue(Issues, TEXT("FixedStepSeconds")));
    TestTrue(TEXT("NaN condition identified"), HasIssue(Issues, TEXT("Conditions.VisibilityFraction")));
    Config = ValidScenario();
    Config.InitialSensors[0].Model.DetectionProbabilityPerSample = 1.1;
    Config.InitialSensors[0].SampleEveryNSteps = 0;
    Config.InitialSensors[0].Transform.SetScale3D(FVector(2));
    TestFalse(TEXT("Invalid sensor rejected"), UIstanaSimulationValidation::ValidateScenario(Config, Issues));
    TestTrue(TEXT("Probability identified"), HasIssue(Issues, TEXT("InitialSensors[0].Model.DetectionProbabilityPerSample")));
    TestTrue(TEXT("Sampling interval identified"), HasIssue(Issues, TEXT("InitialSensors[0].SampleEveryNSteps")));
    TestTrue(TEXT("Scaled sensor identified"), HasIssue(Issues, TEXT("InitialSensors[0].Transform.Scale3D")));
    TestTrue(TEXT("Later success clears old errors"), UIstanaSimulationValidation::ValidateScenario(ValidScenario(), Issues));
    TestEqual(TEXT("No stale errors"), Issues.Num(), 0);
    Config = UIstanaSimulationValidation::MakeExampleScenario();
    TestFalse(TEXT("Example requires a real map selection"), UIstanaSimulationValidation::ValidateScenario(Config, Issues));
    TestTrue(TEXT("Missing map identified"), HasIssue(Issues, TEXT("Map")));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIstanaRuntimeValidationTest, "Istana.Simulation.Contracts.RuntimeData",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FIstanaRuntimeValidationTest::RunTest(const FString& Parameters)
{
    TArray<FIstanaValidationIssue> Issues;
    FIstanaDroneState State;
    State.DroneId = 0;
    State.GroupId = 0;
    TestTrue(TEXT("Valid drone"), UIstanaSimulationValidation::ValidateDroneStates(ValidScenario(), {State}, Issues));
    State.GroupId = 999;
    State.PositionCm.X = std::numeric_limits<double>::infinity();
    TestFalse(TEXT("Unknown group and infinity rejected"), UIstanaSimulationValidation::ValidateDroneStates(ValidScenario(), {State, State}, Issues));
    TestTrue(TEXT("Unknown group identified"), HasIssue(Issues, TEXT("States[0].GroupId")));
    TestTrue(TEXT("Invalid coordinate identified"), HasIssue(Issues, TEXT("States[0].PositionCm")));
    TestTrue(TEXT("Duplicate drone identified"), HasIssue(Issues, TEXT("States[1].DroneId")));

    FIstanaDetectionEvent Event;
    Event.SensorId = 0;
    Event.Observation.bHasPosition = true;
    Event.Confidence = 0;
    TestTrue(TEXT("Zero confidence is valid"), UIstanaSimulationValidation::ValidateDetection(Event, {0}, Issues));
    Event.SampleStep = 2;
    Event.DeliveryStep = 1;
    Event.Observation.bHasPosition = false;
    TestFalse(TEXT("Invalid detection rejected"), UIstanaSimulationValidation::ValidateDetection(Event, {1}, Issues));
    TestTrue(TEXT("Unknown sensor identified"), HasIssue(Issues, TEXT("SensorId")));
    TestTrue(TEXT("Backwards latency identified"), HasIssue(Issues, TEXT("DeliveryStep")));
    TestTrue(TEXT("Missing measurement identified"), HasIssue(Issues, TEXT("Observation.bHasPosition")));

    FIstanaEpisodeResult Result;
    Result.RunId = FGuid::NewGuid();
    Result.ScenarioId = TEXT("Test");
    Result.SimulationVersion = TEXT("test-1");
    Result.SensorModelVersion = TEXT("abstract-test-1");
    Result.CompletionReason = EIstanaCompletionReason::Cancelled;
    TestTrue(TEXT("Cancelled empty episode is valid"), UIstanaSimulationValidation::ValidateEpisodeResult(Result, Issues));
    Result.ExecutedSteps = 1;
    Result.Metrics.SensorSampleCount = 1;
    Result.Metrics.EligibleObjectSampleCount = 1;
    Result.Metrics.TrueDetectionCount = 1;
    Result.Metrics.bHasFirstDetection = true;
    TestTrue(TEXT("First detection at zero is valid"), UIstanaSimulationValidation::ValidateEpisodeResult(Result, Issues));
    Result.Metrics.FirstDetectionStep = 1;
    Result.Metrics.MissedDetectionCount = 1;
    TestFalse(TEXT("Inconsistent result rejected"), UIstanaSimulationValidation::ValidateEpisodeResult(Result, Issues));
    TestTrue(TEXT("Counts checked"), HasIssue(Issues, TEXT("Metrics.EligibleObjectSampleCount")));
    TestTrue(TEXT("Detection outside episode checked"), HasIssue(Issues, TEXT("Metrics.FirstDetectionStep")));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIstanaPresetIsolationTest, "Istana.Simulation.Contracts.PresetIsolation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FIstanaPresetIsolationTest::RunTest(const FString& Parameters)
{
    UIstanaScenarioDataAsset* Preset = NewObject<UIstanaScenarioDataAsset>();
    Preset->Config = ValidScenario();
    FIstanaScenarioConfig Runtime = Preset->CreateRuntimeConfig();
    Runtime.Swarms[0].DroneCount = 100;
    Runtime.InitialSensors[0].Model.CoverageRadiusCm = 99;
    TestEqual(TEXT("Nested swarm copy isolated"), Preset->Config.Swarms[0].DroneCount, 3);
    TestEqual(TEXT("Nested model copy isolated"), Preset->Config.InitialSensors[0].Model.CoverageRadiusCm, 1000.0);
    UIstanaSensorPresetDataAsset* SensorPreset = NewObject<UIstanaSensorPresetDataAsset>();
    FIstanaSensorConfig Sensor = SensorPreset->CreateSensorConfig(7, FTransform(FVector(10, 20, 30)));
    TestEqual(TEXT("Instance ID applied"), Sensor.SensorId, 7);
    TestTrue(TEXT("Placement applied"), Sensor.Transform.GetLocation().Equals(FVector(10, 20, 30)));
    Sensor.Model.CoverageRadiusCm = 8;
    TestEqual(TEXT("Sensor defaults isolated"), SensorPreset->Model.CoverageRadiusCm, 1000.0);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FIstanaPolicyLifecycleTest, "Istana.Simulation.Contracts.PolicyLifecycle",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FIstanaPolicyLifecycleTest::RunTest(const FString& Parameters)
{
    UIstanaNoOpPolicy* Policy = NewObject<UIstanaNoOpPolicy>();
    FIstanaPolicyContext Context;
    Context.RunId = FGuid::NewGuid();
    TArray<FIstanaValidationIssue> Issues;
    IIstanaPolicyInterface::Execute_ResetPolicy(Policy, Context);
    TestFalse(TEXT("Cannot act before observations"), UIstanaSimulationValidation::ValidateActionBatch(
        IIstanaPolicyInterface::Execute_ProduceActions(Policy), Context, 0, Issues));
    FIstanaPolicyObservationBatch Observations;
    Observations.RunId = Context.RunId;
    Observations.DecisionStep = 4;
    IIstanaPolicyInterface::Execute_ReceiveObservations(Policy, Observations);
    TestTrue(TEXT("Reflected interface produces valid no-op"), UIstanaSimulationValidation::ValidateActionBatch(
        IIstanaPolicyInterface::Execute_ProduceActions(Policy), Context, 4, Issues));
    Observations.DecisionStep = 2;
    IIstanaPolicyInterface::Execute_ReceiveObservations(Policy, Observations);
    TestEqual(TEXT("Stale step ignored"), IIstanaPolicyInterface::Execute_ProduceActions(Policy).DecisionStep, int64(4));
    Observations.RunId = FGuid::NewGuid();
    Observations.DecisionStep = 10;
    IIstanaPolicyInterface::Execute_ReceiveObservations(Policy, Observations);
    TestEqual(TEXT("Other episode ignored"), IIstanaPolicyInterface::Execute_ProduceActions(Policy).DecisionStep, int64(4));
    const FIstanaPolicyActionBatch OldActions = IIstanaPolicyInterface::Execute_ProduceActions(Policy);
    Context.RunId = FGuid::NewGuid();
    IIstanaPolicyInterface::Execute_ResetPolicy(Policy, Context);
    TestEqual(TEXT("Reset clears last step"), IIstanaPolicyInterface::Execute_ProduceActions(Policy).DecisionStep, int64(INDEX_NONE));
    TestFalse(TEXT("Old episode actions rejected"), UIstanaSimulationValidation::ValidateActionBatch(OldActions, Context, 4, Issues));
    return true;
}
#endif
