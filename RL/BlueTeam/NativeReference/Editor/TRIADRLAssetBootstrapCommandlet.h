#pragma once

#include "Commandlets/Commandlet.h"
#include "TRIADRLAssetBootstrapCommandlet.generated.h"

/** Creates the source-controlled default RL definition at /Game/TRIAD/RL without overwriting it. */
UCLASS()
class UTRIADRLAssetBootstrapCommandlet : public UCommandlet
{
    GENERATED_BODY()

public:
    UTRIADRLAssetBootstrapCommandlet();
    virtual int32 Main(const FString& Params) override;
};
