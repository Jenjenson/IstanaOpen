using UnrealBuildTool;

public class IstanaOpen : ModuleRules
{
    public IstanaOpen(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        // Shared contracts are included as Simulation/... throughout this module.
        PublicIncludePaths.Add(ModuleDirectory);
        PublicDependencyModuleNames.AddRange(new[] { "Core", "CoreUObject", "Engine", "InputCore" });
        if (Target.bBuildEditor) PrivateDependencyModuleNames.Add("UnrealEd");
        PrivateDependencyModuleNames.AddRange(new[] { "Json", "JsonUtilities", "Sockets", "RHI" });
    }
}
