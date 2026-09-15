#pragma once
#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "Simulation/RedTeam/RedTeamPlacementTypes.h"
#include "RedTeamPlacementPolicy.generated.h"

UINTERFACE(BlueprintType)
class ISTANAOPEN_API URedTeamPlacementPolicy : public UInterface { GENERATED_BODY() };
class ISTANAOPEN_API IRedTeamPlacementPolicy
{
    GENERATED_BODY()
public:
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category="Red Team|Policy")
    void ResetPlacementPolicy(const FRedTeamPlacementContext& Context);
    // May return immediately and submit asynchronously through the manager later.
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category="Red Team|Policy")
    void RequestPlacement(const FRedTeamPlacementContext& Context);
};

/** Deterministic example provider: externally supplied centers, no hidden random layout. */
UCLASS(BlueprintType, Blueprintable)
class ISTANAOPEN_API URedTeamScriptedPlacementPolicy : public UObject, public IRedTeamPlacementPolicy
{
    GENERATED_BODY()
public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Red Team") TObjectPtr<class ARedTeamManager> Manager;
    virtual void ResetPlacementPolicy_Implementation(const FRedTeamPlacementContext& Context) override {}
    virtual void RequestPlacement_Implementation(const FRedTeamPlacementContext& Context) override;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Red Team") TArray<FRedTeamPlacementCenter> Centers;
    UFUNCTION(BlueprintCallable, Category="Red Team")
    FRedTeamPlacementAction MakeAction(const FRedTeamPlacementContext& Context, int64 RequestId) const
    {
        FRedTeamPlacementAction Action;
        Action.RunId = Context.RunId; Action.Revision = Context.Revision;
        Action.RequestId = RequestId; Action.Centers = Centers;
        return Action;
    }
};
