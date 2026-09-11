using UnrealBuildTool;

public class IstanaOpen : ModuleRules
{
    public IstanaOpen(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        // Shared contracts are included as Simulation/... throughout this module.
        PublicIncludePaths.Add(ModuleDirectory);
        PublicDependencyModuleNames.AddRange(new[] { "Core", "CoreUObject", "Engine", "InputCore" });
        PrivateDependencyModuleNames.AddRange(new[] { "Json", "RHI" });
    }
}
