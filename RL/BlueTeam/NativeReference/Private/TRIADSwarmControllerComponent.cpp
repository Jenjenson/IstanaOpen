#include "TRIADSwarmControllerComponent.h"

#include "CesiumGeoreference.h"
#include "CesiumGlobeAnchorComponent.h"
#include "Engine/World.h"
#include "TRIADRLTrainingModel.h"

UTRIADSwarmControllerComponent::UTRIADSwarmControllerComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void UTRIADSwarmControllerComponent::Configure(
    ACesiumGeoreference* InGeoreference,
    double InLongitudeDegrees,
    double InLatitudeDegrees,
    double InHeightMeters,
    double InMaximumSpeedMetersPerSecond)
{
    Georeference = InGeoreference;
    GlobeAnchor = GetOwner() ? GetOwner()->FindComponentByClass<UCesiumGlobeAnchorComponent>() : nullptr;
    LongitudeDegrees = InLongitudeDegrees;
    LatitudeDegrees = InLatitudeDegrees;
    HeightMeters = InHeightMeters;
    MaximumSpeedMetersPerSecond = FMath::Clamp(InMaximumSpeedMetersPerSecond, 0.1, 50.0);
    VelocityEnuMetersPerSecond = FVector::ZeroVector;
    PublishPosition();
}

bool UTRIADSwarmControllerComponent::ApplyNormalizedVelocity(
    const FVector& NormalizedVelocityEnu, double FixedStepSeconds)
{
    if (!FMath::IsFinite(NormalizedVelocityEnu.X) ||
        !FMath::IsFinite(NormalizedVelocityEnu.Y) ||
        !FMath::IsFinite(NormalizedVelocityEnu.Z) ||
        !FMath::IsFinite(FixedStepSeconds) || FixedStepSeconds <= 0.0)
    {
        VelocityEnuMetersPerSecond = FVector::ZeroVector;
        return false;
    }

    const FVector Action = FTRIADRLTrainingModel::ClampNormalizedAction(NormalizedVelocityEnu);
    VelocityEnuMetersPerSecond = Action.GetClampedToMaxSize(1.0) * MaximumSpeedMetersPerSecond;
    const FVector DeltaEnuMeters = VelocityEnuMetersPerSecond * FixedStepSeconds;
    constexpr double Wgs84EquatorialRadiusMeters = 6378137.0;
    const double LatitudeRadians = FMath::DegreesToRadians(LatitudeDegrees);
    const double CosLatitude = FMath::Max(FMath::Abs(FMath::Cos(LatitudeRadians)), 0.000001);
    const double NextLatitude = LatitudeDegrees +
        FMath::RadiansToDegrees(DeltaEnuMeters.Y / Wgs84EquatorialRadiusMeters);
    const double NextLongitude = LongitudeDegrees +
        FMath::RadiansToDegrees(DeltaEnuMeters.X / (Wgs84EquatorialRadiusMeters * CosLatitude));
    const double NextHeight = HeightMeters + DeltaEnuMeters.Z;

    if (!FMath::IsFinite(NextLongitude) || !FMath::IsFinite(NextLatitude) || !FMath::IsFinite(NextHeight) ||
        NextHeight < -100.0 || NextHeight > 1000.0)
    {
        VelocityEnuMetersPerSecond = FVector::ZeroVector;
        return false;
    }

    if (Georeference && GetWorld() && GetOwner())
    {
        const FVector NextWorld = Georeference->TransformLongitudeLatitudeHeightPositionToUnreal(
            FVector(NextLongitude, NextLatitude, NextHeight));
        FHitResult Hit;
        FCollisionQueryParams QueryParams(SCENE_QUERY_STAT(TRIADRLSwarmMotion), true, GetOwner());
        if (GetWorld()->LineTraceSingleByChannel(
                Hit, GetOwner()->GetActorLocation(), NextWorld, ECC_Visibility, QueryParams))
        {
            VelocityEnuMetersPerSecond = FVector::ZeroVector;
            return false;
        }
    }

    LongitudeDegrees = NextLongitude;
    LatitudeDegrees = NextLatitude;
    HeightMeters = NextHeight;
    PublishPosition();
    return true;
}

void UTRIADSwarmControllerComponent::PublishPosition()
{
    if (GlobeAnchor)
    {
        GlobeAnchor->MoveToLongitudeLatitudeHeight(FVector(LongitudeDegrees, LatitudeDegrees, HeightMeters));
    }
}
