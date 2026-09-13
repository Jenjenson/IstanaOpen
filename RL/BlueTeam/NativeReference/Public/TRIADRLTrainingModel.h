#pragma once

#include "CoreMinimal.h"
#include "TRIADRLTrainingTypes.h"

/** Pure deterministic math for the simulation-only RL episode layer. */
class TRIADSENSORFUSION_API FTRIADRLTrainingModel
{
public:
    static bool ValidateConfig(const FTRIADRLTrainingConfig& Config, FString& OutError);
    static int32 GetSensorModalityMask(const FTRIADGeodeticSensorNode& Sensor);
    static double GreatCircleDistanceMeters(double LongitudeA, double LatitudeA, double LongitudeB, double LatitudeB);
    static FVector ClampNormalizedAction(const FVector& Action);
    static bool AdvanceTrack(FTRIADRLTrackState& Track, bool bDetected, int32 StepIndex, int32 ConfirmationSteps);
    static void ComputeStepRewards(
        const FTRIADRLTrainingConfig& Config,
        double PreviousZoneDistanceMeters,
        double CurrentZoneDistanceMeters,
        double NewlyDetectedFraction,
        double EarlyDetectionQuality,
        double TrackedFraction,
        ETRIADRLTerminationReason TerminalReason,
        FTRIADRLRewardBreakdown& OutRewards);
};
