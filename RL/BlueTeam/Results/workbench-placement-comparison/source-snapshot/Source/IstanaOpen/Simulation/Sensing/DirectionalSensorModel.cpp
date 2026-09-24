#include "Simulation/Sensing/DirectionalSensorModel.h"

namespace
{
    bool FiniteInRange(double Value, double Low, double High)
    { return FMath::IsFinite(Value) && Value >= Low && Value <= High; }
}

double IstanaDirectionalSensor::VerticalFovFromHorizontal(double HorizontalFovDegrees,
    int32 ResolutionX, int32 ResolutionY)
{
    if (!FiniteInRange(HorizontalFovDegrees, .01, 179.) || ResolutionX <= 0 || ResolutionY <= 0) return 0;
    const double HalfHorizontal = FMath::DegreesToRadians(HorizontalFovDegrees * .5);
    return FMath::RadiansToDegrees(2. * FMath::Atan(FMath::Tan(HalfHorizontal) * ResolutionY / ResolutionX));
}

bool IstanaDirectionalSensor::IsOrientationBin(const FDirectionalSensorProfile& Profile,
    double YawDegrees, double PitchDegrees)
{
    auto Contains = [](const TArray<double>& Values, double Value)
    { return Values.ContainsByPredicate([&](double Candidate) { return FMath::IsNearlyEqual(Candidate, Value, 1.e-7); }); };
    return Contains(Profile.YawBinsDegrees, YawDegrees) && Contains(Profile.PitchBinsDegrees, PitchDegrees);
}

FDirectionalSensorLook IstanaDirectionalSensor::Evaluate(const FDirectionalSensorProfile& Profile,
    const FVector& SensorWorldCm, const FRotator& Orientation, const FVector& TargetWorldCm,
    double TargetSizeM, double BaseProbability, double Visibility, double Rain, double Humidity,
    bool bHasLineOfSight)
{
    FDirectionalSensorLook Result;
    const FVector OffsetCm = TargetWorldCm - SensorWorldCm;
    Result.DistanceM = OffsetCm.Size() / 100.;
    Result.bHasLineOfSight = bHasLineOfSight;
    if (!Profile.bEnabled || Result.DistanceM <= SMALL_NUMBER || !FMath::IsFinite(TargetSizeM)) return Result;

    const FVector Local = Orientation.UnrotateVector(OffsetCm.GetSafeNormal());
    Result.bInFront = Local.X > 0;
    Result.HorizontalOffsetDegrees = FMath::RadiansToDegrees(FMath::Atan2(Local.Y, Local.X));
    Result.VerticalOffsetDegrees = FMath::RadiansToDegrees(FMath::Atan2(Local.Z,
        FMath::Sqrt(Local.X * Local.X + Local.Y * Local.Y)));
    const double HalfH = Profile.HorizontalFovDegrees * .5;
    const double HalfV = Profile.VerticalFovDegrees * .5;
    Result.bInsideFov = Result.bInFront && FMath::Abs(Result.HorizontalOffsetDegrees) <= HalfH
        && FMath::Abs(Result.VerticalOffsetDegrees) <= HalfV;
    Result.bWithinEvaluationDistance = Result.DistanceM <= Profile.MaxEvaluationDistanceM;

    const double AngularSizeRadians = 2. * FMath::Atan(FMath::Max(0., TargetSizeM) / (2. * Result.DistanceM));
    const double IfovRadians = Profile.IfovMilliradians / 1000.;
    Result.PixelsOnTarget = IfovRadians > 0 ? AngularSizeRadians / IfovRadians : 0;
    if (!Result.bInsideFov || !Result.bWithinEvaluationDistance
        || (Profile.bRequireLineOfSight && !Result.bHasLineOfSight)) return Result;

    // All factors below are simulator assumptions, not Teledyne FLIR ratings.
    const double PixelResponse = 1. - FMath::Exp(-Result.PixelsOnTarget / Profile.PixelsFor63Percent);
    const double NoiseEquivalentK = Profile.NedtMillikelvin / 1000.;
    const double ContrastResponse = Profile.NominalThermalContrastK
        / (Profile.NominalThermalContrastK + Profile.ContrastNoiseMultiplier * NoiseEquivalentK);
    const double NormalizedEdge = FMath::Max(FMath::Abs(Result.HorizontalOffsetDegrees) / HalfH,
        FMath::Abs(Result.VerticalOffsetDegrees) / HalfV);
    const double EdgeResponse = FMath::Pow(FMath::Max(0., FMath::Cos(NormalizedEdge * PI * .5)),
        Profile.EdgeFalloffExponent);
    const double RangeTaper = FMath::Square(FMath::Max(0., FMath::Cos(
        Result.DistanceM / Profile.MaxEvaluationDistanceM * PI * .5)));
    const double Atmosphere = FMath::Exp(-Result.DistanceM / Profile.AtmosphericAttenuationDistanceM);
    const double Weather = FMath::Clamp(Visibility * (1. - Profile.RainLossAtMaximum * Rain)
        * (1. - Profile.HumidityLossAtMaximum * Humidity), 0., 1.);
    Result.DetectionProbability = FMath::Clamp(BaseProbability * PixelResponse * ContrastResponse
        * EdgeResponse * RangeTaper * Atmosphere * Weather, 0., 1.);
    return Result;
}
