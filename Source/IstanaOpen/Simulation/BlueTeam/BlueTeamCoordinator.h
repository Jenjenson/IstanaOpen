#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Simulation/RedTeam/RedTeamPlacementTypes.h"
#include "Simulation/Sensing/DirectionalSensorModel.h"
#include "BlueTeamCoordinator.generated.h"

class ARedTeamManager;
class FJsonObject;
class USceneComponent;
class UStaticMeshComponent;
class UStaticMesh;
class UMaterialInterface;

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
    // Optional generic directional capability. Non-directional profiles retain
    // the legacy radial model; Boson+ is the first configured real profile.
    UPROPERTY(EditAnywhere, BlueprintReadWrite) FDirectionalSensorProfile Directional;
};

USTRUCT(BlueprintType)
struct ISTANAOPEN_API FBlueSensorPlacement
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere, BlueprintReadWrite) FString ProfileId;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) int32 SiteId = INDEX_NONE;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) double YawDegrees = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite) double PitchDegrees = 0;
};

/** Visible presentation of accepted sensors; no collision and no independent sensing clock. */
UCLASS()
class ISTANAOPEN_API ABlueSensorMarker : public AActor
{
    GENERATED_BODY()
public:
    ABlueSensorMarker();
    void SetMastHeight(double HeightM);
    void ConfigureSensor(const FString& ProfileId, double HeightM);
    void ConfigureOrientation(double YawDegrees, double PitchDegrees);
private:
    UStaticMeshComponent* Part(UStaticMesh* Mesh, UMaterialInterface* Material,
        const FVector& Position, const FVector& SizeCm, const FRotator& Rotation = FRotator::ZeroRotator,
        USceneComponent* Parent = nullptr);
    void Strut(const FVector& A, const FVector& B, double DiameterCm);
    UPROPERTY() TObjectPtr<UStaticMesh> CubeMesh;
    UPROPERTY() TObjectPtr<UStaticMesh> CylinderMesh;
    UPROPERTY() TObjectPtr<UStaticMesh> SphereMesh;
    UPROPERTY() TObjectPtr<UMaterialInterface> MetalMaterial;
    UPROPERTY() TObjectPtr<UMaterialInterface> PaintMaterial;
    UPROPERTY() TObjectPtr<UMaterialInterface> LensMaterial;
    UPROPERTY() TObjectPtr<UMaterialInterface> RubberMaterial;
    UPROPERTY() TObjectPtr<USceneComponent> OpticalHead;
    UPROPERTY(Transient) TArray<TObjectPtr<UStaticMeshComponent>> Parts;
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
    // -1 draws every placed sensor; otherwise draws one selected placement's
    // frustum and per-target diagnostics to keep the view readable.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Blue Team|Presentation") int32 DebugSelectedSensorIndex = 0;
    // Observer-only capture overlays; never part of the public policy snapshot.
    UPROPERTY(Transient) bool bCapturePresentation = false;

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
        double FirstDetectionPixels = 0, FirstDetectionProbability = 0;
        double LastDistanceM = 0, LastPixelsOnTarget = 0, LastDetectionProbability = 0;
        bool bLastInsideFov = false, bLastLineOfSight = false, bLastBlockedByGeometry = false;
        FString LastSensorId;
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
    FDirectionalSensorLook DetectionLook(const FBlueSensorProfile& Profile, const FBlueSensorPlacement& Placement,
        const FVector& Target, double TargetSizeM, bool bEmitting) const;
    bool HasLineOfSight(const FVector& Sensor, const FVector& Target) const;
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
