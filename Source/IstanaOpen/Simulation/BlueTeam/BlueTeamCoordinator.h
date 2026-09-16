#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Simulation/RedTeam/RedTeamPlacementTypes.h"
#include "BlueTeamCoordinator.generated.h"

class ARedTeamManager;
class FJsonObject;
class UStaticMeshComponent;

/** Synthetic capabilities in metres. These are not calibrated physical sensors. */
USTRUCT(BlueprintType)
struct ISTANAOPEN_API FBlueSensorProfile
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite) FString Id;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) FString Label;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) double Cost = 1;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) double HeightM = 4;
    // x=RF, y=radar, z=EO, w=thermal.
    UPROPERTY(EditAnywhere, BlueprintReadWrite) FVector4 RangesM = FVector4(0, 100, 0, 0);
    UPROPERTY(EditAnywhere, BlueprintReadWrite) FVector4 Strengths = FVector4(0, .86, 0, 0);
};

USTRUCT(BlueprintType)
struct ISTANAOPEN_API FBlueSensorPlacement
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite) FString ProfileId;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) int32 SiteId = INDEX_NONE;
};

/** Visible presentation of accepted sensors; no collision and no independent sensing clock. */
UCLASS()
class ISTANAOPEN_API ABlueSensorMarker : public AActor
{
    GENERATED_BODY()
public:
    ABlueSensorMarker();
    void SetMastHeight(double HeightM);
private:
    UPROPERTY() TObjectPtr<UStaticMeshComponent> Body;
};

/** Optional initial-layout Blue evaluator using the Red manager's one fixed-step clock. */
UCLASS(BlueprintType, Blueprintable)
class ISTANAOPEN_API ABlueTeamCoordinator : public AActor
{
    GENERATED_BODY()
public:
    ABlueTeamCoordinator();
    UPROPERTY(EditInstanceOnly, BlueprintReadWrite, Category="Blue Team") TObjectPtr<ARedTeamManager> Manager;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Layout") TArray<FBlueSensorProfile> Catalogue;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Layout") TArray<FVector2D> ApprovedSitesM;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Layout") TArray<int32> BlockedSites;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Layout") TArray<FString> AvailableSensorIds;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Layout") double Budget = 3;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Layout") int32 MaxSites = 3;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Layout") double MinimumSeparationM = 20;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Layout") double DeploymentMinRadiusM = 30;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Layout") double DeploymentMaxRadiusM = 150;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Mission") double ObjectiveRadiusM = 20;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Mission") double DefenceLeadTimeSeconds = 4;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Mission") double LookIntervalSeconds = 1;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Mission") int32 RequiredConfirmations = 2;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Mission") int32 ConfirmationWindow = 3;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Mission") double TimeLimitSeconds = 96;
    // Public, explicitly configured beliefs; never filled from private Red state.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Public priors") double PriorSpawnRadiusM = 320;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Public priors") double PriorAltitudeM = 45;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Public priors") double PriorSpeedMps = 12;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Public priors") double PriorEmitterProbability = .5;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Public priors") int32 PriorSwarmSize = 3;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Public priors") TArray<double> ApproachWeights;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Weather") double Visibility = 1;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Weather") double Rain = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Weather") double Illumination = 1;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Weather") double Humidity = .4;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Weather") double RFNoise = .1;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Presentation") bool bSpawnSensorMarkers = true;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Presentation") bool bDrawCoverage = true;

    bool ValidateConfiguration(double FixedStep, FString& Error) const;
    void BeginEpisode(const FRedTeamPlacementContext& Context);
    void CaptureInitialStates(const TArray<FIstanaDroneState>& States);
    bool CanAdvance(FString& Error) const;
    void Evaluate(FRedTeamEpisodeObservation& Observation);
    TSharedRef<FJsonObject> ContextJson() const;
    TSharedRef<FJsonObject> ObservationJson() const;
    TSharedRef<FJsonObject> DeployJson(const FJsonObject& Action, FString& Error);
    bool IsCommitted() const { return bCommitted; }
    FVector SiteWorldCm(int32 SiteId, const FBlueSensorProfile& Profile) const;
    virtual void Tick(float DeltaSeconds) override;
    virtual void Destroyed() override;
    virtual void EndPlay(const EEndPlayReason::Type Reason) override;

private:
    struct FTargetEvidence
    {
        FVector PreviousPosition = FVector::ZeroVector;
        TArray<int64> HitLooks;
        double FirstDetection = -1, FirstConfirmation = -1, ZoneEntry = -1, ReportTime = -1;
        FVector ReportPositionM = FVector::ZeroVector, ReportVelocityMps = FVector::ZeroVector;
        int32 PublicTrackId = INDEX_NONE;
        bool bHasPrevious = false;
    };
    TSharedRef<FJsonObject> PublicSnapshotJson() const;
    FString ConfigurationSignature() const;
    void ClearMarkers();
    bool ResolveSurfaceCm(int32 SiteId, FVector& Surface) const;
    bool SurfaceUnchanged(int32 SiteId) const;
    TArray<FVector> SiteSurfacesCm;
    TArray<bool> SupportedSites;
    double DetectionProbability(const FBlueSensorProfile& Profile, const FVector& Sensor, const FVector& Target, bool bEmitting) const;
    double Uniform(int32 DroneId, int32 SensorId, int64 Look, uint32 Salt) const;
    TMap<int32, FTargetEvidence> Evidence;
    TArray<FBlueSensorPlacement> Placements;
    UPROPERTY(Transient) TArray<TObjectPtr<ABlueSensorMarker>> Markers;
    FGuid RunId;
    int64 Revision = 0, CompletedSteps = 0, LastActionId = -1;
    int32 EpisodeSeed = 0, LookSteps = 1, NextTrackId = 0;
    double FixedStepSeconds = .05, Cost = 0;
    FVector OriginWorldCm = FVector::ZeroVector;
    FString FrozenConfiguration, LastActionPayload;
    TSharedPtr<FJsonObject> LastActionResult;
    bool bCommitted = false;
};
