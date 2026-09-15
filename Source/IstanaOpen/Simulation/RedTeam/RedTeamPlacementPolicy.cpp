#include "Simulation/RedTeam/RedTeamPlacementPolicy.h"
#include "Simulation/RedTeam/RedTeamManager.h"
void URedTeamScriptedPlacementPolicy::RequestPlacement_Implementation(const FRedTeamPlacementContext& Context)
{
    if (IsValid(Manager)) Manager->SubmitPlacement(MakeAction(Context, 0));
}
