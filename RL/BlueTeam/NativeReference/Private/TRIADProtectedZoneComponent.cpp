#include "TRIADProtectedZoneComponent.h"

#include "TRIADRLTrainingModel.h"

UTRIADProtectedZoneComponent::UTRIADProtectedZoneComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void UTRIADProtectedZoneComponent::Configure(const FTRIADRLProtectedZoneDefinition& InDefinition)
{
    Definition = InDefinition;
}

double UTRIADProtectedZoneComponent::SignedHorizontalDistanceMeters(
    double LongitudeDegrees, double LatitudeDegrees) const
{
    return FTRIADRLTrainingModel::GreatCircleDistanceMeters(
        LongitudeDegrees,
        LatitudeDegrees,
        Definition.LongitudeDegrees,
        Definition.LatitudeDegrees) - FMath::Max(Definition.RadiusMeters, 10.0);
}
