#pragma once
#include "CoreMinimal.h"

// Synthetic target-zone arrival, not a physical impact or interception model.
// Unknown arrival remains unresolved; callers must not silently drop such targets.
inline double BlueWarningSeconds(double FirstDetection, double ZoneEntry)
{
    return FirstDetection >= 0 && ZoneEntry >= 0 ? FMath::Max(0., ZoneEntry - FirstDetection) : 0.;
}
