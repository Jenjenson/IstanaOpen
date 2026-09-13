#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "TRIADRLTrainingModel.h"

#include <limits>

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FTRIADRLTrainingModelDeterminismTest,
    "TRIAD.SensorFusion.RL.DeterministicMath",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FTRIADRLTrainingModelDeterminismTest::RunTest(const FString& Parameters)
{
    const double A = FTRIADRLTrainingModel::GreatCircleDistanceMeters(103.842, 1.307, 103.848, 1.307);
    const double B = FTRIADRLTrainingModel::GreatCircleDistanceMeters(103.842, 1.307, 103.848, 1.307);
    TestEqual(TEXT("Same geodesic inputs are deterministic"), A, B);
    TestTrue(TEXT("Distance is finite and positive"), FMath::IsFinite(A) && A > 0.0);
    TestEqual(TEXT("Non-finite action fails closed"),
        FTRIADRLTrainingModel::ClampNormalizedAction(FVector(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0)),
        FVector::ZeroVector);
    TestEqual(TEXT("Action X is clamped"),
        FTRIADRLTrainingModel::ClampNormalizedAction(FVector(2.0, -2.0, 0.5)),
        FVector(1.0, -1.0, 0.5));
    FTRIADGeodeticSensorNode Modalities;
    Modalities.DetectionRangeMeters = 0;
    Modalities.SupportedFrequenciesGHz.Reset();
    Modalities.bEnableSearchRadar = true;
    Modalities.bEnableEOPTZ = false;
    Modalities.bEnableThermalPTZ = true;
    TestEqual(TEXT("Combined profiles expose every enabled modality"),
        FTRIADRLTrainingModel::GetSensorModalityMask(Modalities),
        static_cast<int32>(ETRIADRLSensorModality::SearchRadar) |
            static_cast<int32>(ETRIADRLSensorModality::Thermal));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FTRIADRLTrainingRewardTest,
    "TRIAD.SensorFusion.RL.RewardContract",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FTRIADRLTrainingRewardTest::RunTest(const FString& Parameters)
{
    FTRIADRLTrainingConfig Config;
    Config.MaximumDistanceFromZoneMeters = 300;
    FTRIADRLRewardBreakdown Reward;
    FTRIADRLTrainingModel::ComputeStepRewards(Config, 200.0, 190.0, 1, 0.8, 0, ETRIADRLTerminationReason::None, Reward);
    TestTrue(TEXT("A new detection benefits Blue"), Reward.BlueTotal() > 0.0);
    TestTrue(TEXT("Detection outweighs small Red progress"), Reward.RedTotal() < 0.0);
    FTRIADRLTrainingModel::ComputeStepRewards(Config, 10.0, 0.0, 0, 0, 0, ETRIADRLTerminationReason::ProtectedZoneReached, Reward);
    TestTrue(TEXT("Geofence entry penalizes Blue"), Reward.BlueTotal() <= -10.0);
    TestTrue(TEXT("Geofence entry rewards Red"), Reward.RedTotal() > 9.0);
    for (auto Reason : {ETRIADRLTerminationReason::SustainedTrackDefence, ETRIADRLTerminationReason::HorizonReached,
                        ETRIADRLTerminationReason::ConstraintViolation})
    {
        FTRIADRLTrainingModel::ComputeStepRewards(Config, 100, 100, 0, 0, 0, Reason, Reward);
        TestEqual(TEXT("Every defence outcome has explicit terminal reward"), Reward.BlueTerminal, 10.0);
        TestEqual(TEXT("No hover-at-horizon escape from Red loss"), Reward.RedTerminal, -10.0);
    }
    double TotalProgress = 0;
    const double Distances[] = {250, 100, 200, 250};
    for (int32 Index = 1; Index < 4; ++Index)
    {
        FTRIADRLTrainingModel::ComputeStepRewards(Config, Distances[Index-1], Distances[Index], 0, 0, 0,
            ETRIADRLTerminationReason::None, Reward);
        TotalProgress += Reward.RedProgress;
    }
    TestTrue(TEXT("Out-and-back motion cannot farm progress"), FMath::IsNearlyZero(TotalProgress, 1.e-12));
    for (int32 Horizon : {1, 60, 6000})
    {
        Config.EpisodeHorizonSteps = Horizon;
        double TrackingSum = 0;
        for (int32 Step = 0; Step < Horizon; ++Step)
        {
            FTRIADRLTrainingModel::ComputeStepRewards(Config, 100, 100, 0, 0, 1,
                ETRIADRLTerminationReason::None, Reward);
            TrackingSum += Reward.BlueTracking;
        }
        TestTrue(TEXT("Tracking shaping budget is independent of horizon"),
            FMath::IsNearlyEqual(TrackingSum, Config.Rewards.BlueTracking, 1.e-9));
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTRIADRLTrackSemanticsTest,
    "TRIAD.SensorFusion.RL.CurrentTracksAndOnceOnlyDetection",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FTRIADRLTrackSemanticsTest::RunTest(const FString& Parameters)
{
    FTRIADRLTrackState Track;
    TestTrue(TEXT("First detection gives novelty credit"), FTRIADRLTrainingModel::AdvanceTrack(Track, true, 1, 3));
    TestFalse(TEXT("A single hit is not a track"), Track.bTracked);
    TestFalse(TEXT("Repeated evaluation of same step cannot give novelty"), FTRIADRLTrainingModel::AdvanceTrack(Track, true, 1, 3));
    TestEqual(TEXT("Repeated step cannot advance confirmation"), Track.ConsecutiveDetectionSteps, 1);
    FTRIADRLTrainingModel::AdvanceTrack(Track, true, 2, 3);
    FTRIADRLTrainingModel::AdvanceTrack(Track, true, 3, 3);
    TestTrue(TEXT("Consecutive evidence confirms current track"), Track.bTracked);
    FTRIADRLTrainingModel::AdvanceTrack(Track, false, 4, 3);
    TestFalse(TEXT("Lost detection is not currently detected"), Track.bCurrentlyDetected);
    TestFalse(TEXT("Lost detection clears the current track"), Track.bTracked);
    TestTrue(TEXT("Historical detection remains distinct"), Track.bEverDetected);
    TestFalse(TEXT("Reacquisition cannot farm novelty"), FTRIADRLTrainingModel::AdvanceTrack(Track, true, 5, 3));
    TestEqual(TEXT("First detection latency is stable"), Track.FirstDetectionStep, 1);
    TestEqual(TEXT("Track continuity counts actual tracked steps"), Track.TrackedStepCount, 1);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTRIADRLRewardDominanceTest,
    "TRIAD.SensorFusion.RL.TerminalDominanceValidation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FTRIADRLRewardDominanceTest::RunTest(const FString& Parameters)
{
    FTRIADRLTrainingConfig Config;
    Config.ScenarioId = TEXT("test-catalogue");
    Config.ConfirmationNodeCount = 1;
    FTRIADRLSensorCandidate Candidate;
    Candidate.CandidateId = TEXT("test-north");
    Candidate.MountId = TEXT("north");
    Candidate.OffsetEnuMeters = FVector(0, 200, 4);
    Config.SensorCandidates.Add(Candidate);
    FString Error;
    TestTrue(TEXT("Valid terminal-dominant configuration"), FTRIADRLTrainingModel::ValidateConfig(Config, Error));
    Config.Rewards.BlueDetection = 20;
    TestFalse(TEXT("Detection cannot overshadow winning"), FTRIADRLTrainingModel::ValidateConfig(Config, Error));
    Config.Rewards.BlueDetection = 0.5;
    Config.Rewards.RedProgress = 20;
    TestFalse(TEXT("Progress cannot overshadow breach/defence"), FTRIADRLTrainingModel::ValidateConfig(Config, Error));
    Config.Rewards.RedProgress = 1;
    Config.TrackConfirmationSteps = Config.EpisodeHorizonSteps+1;
    TestFalse(TEXT("Impossible confirmation horizon rejected"), FTRIADRLTrainingModel::ValidateConfig(Config, Error));
    Config.TrackConfirmationSteps = 3;
    Config.RedMovementAxisMask = FVector(0, 0, 0);
    TestFalse(TEXT("Completely inactive movement rejected"), FTRIADRLTrainingModel::ValidateConfig(Config, Error));
    Config.RedMovementAxisMask = FVector(0, 1, 0);
    Config.SensorCandidates[0].bAllowDynamicPosition = true;
    Config.SensorCandidates[0].OffsetEnuMeters = FVector(0, 0, 4);
    TestTrue(TEXT("Dynamic sensor profiles do not require a fixed candidate radius"),
        FTRIADRLTrainingModel::ValidateConfig(Config, Error));
    Config.SensorBudgetUnits = 0.5;
    TestFalse(TEXT("Budget must fund the requested confirmation count"),
        FTRIADRLTrainingModel::ValidateConfig(Config, Error));
    return true;
}

#endif
