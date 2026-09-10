using UnrealBuildTool;

public class IstanaOpen : ModuleRules
{
    public IstanaOpen(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new[] { "Core", "CoreUObject", "Engine", "InputCore" });
        PrivateDependencyModuleNames.AddRange(new[] { "Json", "RHI" });
    }
}
