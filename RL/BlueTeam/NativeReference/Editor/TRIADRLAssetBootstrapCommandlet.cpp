#include "TRIADRLAssetBootstrapCommandlet.h"

#include "AssetRegistry/AssetRegistryModule.h"
#include "Misc/PackageName.h"
#include "TRIADAdversarialTrainingManager.h"
#include "TRIADRLTrainingModel.h"
#include "TRIADRLTrainingTypes.h"
#include "UObject/Package.h"
#include "UObject/SavePackage.h"

namespace
{
const TCHAR* RLAssetPackageName = TEXT("/Game/TRIAD/RL/DA_TRIADRLDefault");
const TCHAR* RLAssetObjectPath = TEXT("/Game/TRIAD/RL/DA_TRIADRLDefault.DA_TRIADRLDefault");
}

UTRIADRLAssetBootstrapCommandlet::UTRIADRLAssetBootstrapCommandlet()
{
    IsClient = false;
    IsEditor = true;
    IsServer = false;
    LogToConsole = true;
    ShowErrorCount = true;
}

int32 UTRIADRLAssetBootstrapCommandlet::Main(const FString& Params)
{
    FTRIADRLTrainingConfig Config;
    FString LoadError;
    if (!ATRIADAdversarialTrainingManager::LoadTrainingConfig(Config, LoadError))
    {
        UE_LOG(LogTemp, Error, TEXT("Could not load the JSON-backed TRIAD RL definition: %s"), *LoadError);
        return 3;
    }

    UTRIADRLTrainingDefinition* Asset = LoadObject<UTRIADRLTrainingDefinition>(
        nullptr, RLAssetObjectPath, nullptr, LOAD_NoWarn);
    const bool bRefreshingExistingAsset = Asset != nullptr;
    UPackage* Package = bRefreshingExistingAsset ? Asset->GetPackage() : CreatePackage(RLAssetPackageName);
    if (!Package)
    {
        UE_LOG(LogTemp, Error, TEXT("Could not load or create the TRIAD RL asset package."));
        return 4;
    }
    if (!Asset)
    {
        Asset = NewObject<UTRIADRLTrainingDefinition>(
            Package,
            UTRIADRLTrainingDefinition::StaticClass(),
            TEXT("DA_TRIADRLDefault"),
            RF_Public | RF_Standalone);
        FAssetRegistryModule::AssetCreated(Asset);
    }

    Asset->Modify();
    Asset->Config = Config;
    Package->MarkPackageDirty();

    const FString Filename = FPackageName::LongPackageNameToFilename(
        RLAssetPackageName, FPackageName::GetAssetPackageExtension());
    FSavePackageArgs SaveArgs;
    SaveArgs.TopLevelFlags = RF_Public | RF_Standalone;
    SaveArgs.SaveFlags = SAVE_NoError;
    SaveArgs.Error = GError;
    if (!UPackage::SavePackage(Package, Asset, *Filename, SaveArgs))
    {
        UE_LOG(LogTemp, Error, TEXT("Could not save TRIAD RL asset '%s'."), *Filename);
        return 5;
    }
    UE_LOG(
        LogTemp,
        Display,
        TEXT("%s TRIAD RL asset from the authoritative JSON: %s"),
        bRefreshingExistingAsset ? TEXT("Refreshed") : TEXT("Created"),
        RLAssetObjectPath);
    return 0;
}
