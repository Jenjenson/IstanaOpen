#pragma once

#include "CoreMinimal.h"
#include "DirectionalSensorModel.generated.h"

/**
 * Generic camera-like directional sensor description.
 *
 * The Manufacturer* fields are documentary hardware specifications.  The
 * Simulation* fields are explicit modelling choices and must not be presented
 * as manufacturer-rated detection performance.
 */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FDirectionalSensorProfile
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Manufacturer specification") bool bEnabled = false;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Manufacturer specification") FString Manufacturer;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Manufacturer specification") FString Model;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Manufacturer specification") FString Modality;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Manufacturer specification") FString SpecificationSource;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Manufacturer specification") int32 ResolutionX = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Manufacturer specification") int32 ResolutionY = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Manufacturer specification") double PixelPitchMicrometres = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Manufacturer specification") double HorizontalFovDegrees = 0;
    // Calculated from HFOV and sensor aspect ratio when the manufacturer page
    // does not publish VFOV; serialized separately as a derived value.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Calculated camera geometry") double VerticalFovDegrees = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Manufacturer specification") double IfovMilliradians = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Manufacturer specification") double FrameRateHz = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Manufacturer specification") double SelectableFrameRateHz = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Manufacturer specification") double NedtMillikelvin = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Simulation assumption") FString DetectionModel = TEXT("pixels_on_target_v1");
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Simulation assumption") double MaxEvaluationDistanceM = 500;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Simulation assumption") double NominalThermalContrastK = 8;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Simulation assumption") double ContrastNoiseMultiplier = 8;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Simulation assumption") double PixelsFor63Percent = 3;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Simulation assumption") double AtmosphericAttenuationDistanceM = 900;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Simulation assumption") double RainLossAtMaximum = .45;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Simulation assumption") double HumidityLossAtMaximum = .35;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Simulation assumption") double EdgeFalloffExponent = 1.5;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Simulation assumption") bool bRequireLineOfSight = true;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL orientation") TArray<double> YawBinsDegrees;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="RL orientation") TArray<double> PitchBinsDegrees;
};

/** Deterministic geometry/probability result before the temporal hit draw. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FDirectionalSensorLook
{
    GENERATED_BODY()
    UPROPERTY(BlueprintReadOnly) bool bInFront = false;
    UPROPERTY(BlueprintReadOnly) bool bInsideFov = false;
    UPROPERTY(BlueprintReadOnly) bool bWithinEvaluationDistance = false;
    UPROPERTY(BlueprintReadOnly) bool bHasLineOfSight = false;
    UPROPERTY(BlueprintReadOnly) double DistanceM = 0;
    UPROPERTY(BlueprintReadOnly) double HorizontalOffsetDegrees = 0;
    UPROPERTY(BlueprintReadOnly) double VerticalOffsetDegrees = 0;
    UPROPERTY(BlueprintReadOnly) double PixelsOnTarget = 0;
    UPROPERTY(BlueprintReadOnly) double DetectionProbability = 0;
};

namespace IstanaDirectionalSensor
{
    ISTANAOPEN_API double VerticalFovFromHorizontal(double HorizontalFovDegrees, int32 ResolutionX, int32 ResolutionY);
    ISTANAOPEN_API bool IsOrientationBin(const FDirectionalSensorProfile& Profile, double YawDegrees, double PitchDegrees);
    ISTANAOPEN_API FDirectionalSensorLook Evaluate(const FDirectionalSensorProfile& Profile,
        const FVector& SensorWorldCm, const FRotator& Orientation, const FVector& TargetWorldCm,
        double TargetSizeM, double BaseProbability, double Visibility, double Rain, double Humidity,
        bool bHasLineOfSight);
}
