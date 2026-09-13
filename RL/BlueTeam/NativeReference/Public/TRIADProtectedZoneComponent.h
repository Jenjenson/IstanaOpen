#pragma once

#include "Components/ActorComponent.h"
#include "TRIADRLTrainingTypes.h"
#include "TRIADProtectedZoneComponent.generated.h"

/** Simulation-only geofence; it has no collision, damage, or targeting behavior. */
UCLASS(ClassGroup = (TRIAD), BlueprintType, meta = (BlueprintSpawnableComponent))
class TRIADSENSORFUSION_API UTRIADProtectedZoneComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UTRIADProtectedZoneComponent();

    UFUNCTION(BlueprintCallable, Category = "TRIAD|RL|Protected Zone")
    void Configure(const FTRIADRLProtectedZoneDefinition& InDefinition);

    UFUNCTION(BlueprintPure, Category = "TRIAD|RL|Protected Zone")
    double SignedHorizontalDistanceMeters(double LongitudeDegrees, double LatitudeDegrees) const;

    const FTRIADRLProtectedZoneDefinition& GetDefinition() const { return Definition; }

private:
    UPROPERTY(VisibleAnywhere, Category = "TRIAD|RL|Protected Zone")
    FTRIADRLProtectedZoneDefinition Definition;
};
