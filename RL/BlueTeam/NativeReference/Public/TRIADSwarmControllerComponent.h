#pragma once

#include "Components/ActorComponent.h"
#include "TRIADSwarmControllerComponent.generated.h"

class ACesiumGeoreference;
class UCesiumGlobeAnchorComponent;

/** Applies bounded, normalized simulation-only motion. This component cannot emit vehicle or motor commands. */
UCLASS(ClassGroup = (TRIAD), BlueprintType, meta = (BlueprintSpawnableComponent))
class TRIADSENSORFUSION_API UTRIADSwarmControllerComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UTRIADSwarmControllerComponent();

    void Configure(
        ACesiumGeoreference* InGeoreference,
        double InLongitudeDegrees,
        double InLatitudeDegrees,
        double InHeightMeters,
        double InMaximumSpeedMetersPerSecond);

    /** Returns false for a non-finite action or blocked simulation movement. */
    bool ApplyNormalizedVelocity(const FVector& NormalizedVelocityEnu, double FixedStepSeconds);

    double GetLongitudeDegrees() const { return LongitudeDegrees; }
    double GetLatitudeDegrees() const { return LatitudeDegrees; }
    double GetHeightMeters() const { return HeightMeters; }
    FVector GetVelocityEnuMetersPerSecond() const { return VelocityEnuMetersPerSecond; }

private:
    void PublishPosition();

    UPROPERTY(Transient)
    TObjectPtr<ACesiumGeoreference> Georeference;

    UPROPERTY(Transient)
    TObjectPtr<UCesiumGlobeAnchorComponent> GlobeAnchor;

    double LongitudeDegrees = 0.0;
    double LatitudeDegrees = 0.0;
    double HeightMeters = 0.0;
    double MaximumSpeedMetersPerSecond = 15.0;
    FVector VelocityEnuMetersPerSecond = FVector::ZeroVector;
};
