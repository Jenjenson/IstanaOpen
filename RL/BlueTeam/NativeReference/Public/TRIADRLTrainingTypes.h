#pragma once

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "TRIADSensorFusionTypes.h"
#include "TRIADRLTrainingTypes.generated.h"

UENUM(BlueprintType)
enum class ETRIADRLPhase : uint8
{
    Inactive,
    BluePlacement,
    RedDeployment,
    RedMovement,
    Terminal
};

UENUM(BlueprintType)
enum class ETRIADRLTerminationReason : uint8
{
    None,
    ProtectedZoneReached,
    AllTargetsConfirmed,
    HorizonReached,
    ConstraintViolation,
    InvalidAction,
    SustainedTrackDefence
};

/** Sensor capabilities visible to the placement policy. A profile may combine modalities. */
UENUM(BlueprintType, meta = (Bitflags, UseEnumValuesAsMaskValuesInEditor = "true"))
enum class ETRIADRLSensorModality : uint8
{
    None = 0 UMETA(Hidden),
    PassiveRF = 1 << 0,
    SearchRadar = 1 << 1,
    ElectroOptical = 1 << 2,
    Thermal = 1 << 3
};
ENUM_CLASS_FLAGS(ETRIADRLSensorModality);

USTRUCT(BlueprintType)
struct TRIADSENSORFUSION_API FTRIADRLProtectedZoneDefinition
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Protected Zone")
    double LongitudeDegrees = 103.84288055;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Protected Zone")
    double LatitudeDegrees = 1.30709615;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Protected Zone")
    double HeightMeters = 47.0;

    /** Abstract geofence only. Building contact, damage, and payload delivery are outside this model. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Protected Zone", meta = (ClampMin = "10.0"))
    double RadiusMeters = 150.0;
};

USTRUCT(BlueprintType)
struct TRIADSENSORFUSION_API FTRIADRLRewardWeights
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Reward")
    double BlueDetection = 0.5;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Reward")
    double BlueEarlyDetection = 0.5;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Reward")
    double BlueTracking = 1.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Reward")
    double BlueDefenceWin = 10.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Reward")
    double BlueZoneMiss = -10.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Reward")
    double BlueSiteCost = -0.25;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Reward")
    double RedProgress = 1.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Reward")
    double RedDetected = -0.5;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Reward")
    double RedTracking = -0.5;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Reward")
    double RedTime = -0.1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Reward")
    double RedDefenceLoss = -10.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Reward")
    double RedZoneReached = 10.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Reward")
    double RedConstraintViolation = -10.0;
};

/** Per-transition components, not cumulative team totals. */
USTRUCT(BlueprintType)
struct TRIADSENSORFUSION_API FTRIADRLRewardBreakdown
{
    GENERATED_BODY()
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") double BlueDetection = 0.0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") double BlueEarlyDetection = 0.0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") double BlueTracking = 0.0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") double BlueEfficiency = 0.0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") double BlueTerminal = 0.0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") double RedProgress = 0.0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") double RedDetection = 0.0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") double RedTracking = 0.0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") double RedTime = 0.0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") double RedTerminal = 0.0;
    double BlueTotal() const { return BlueDetection + BlueEarlyDetection + BlueTracking + BlueEfficiency + BlueTerminal; }
    double RedTotal() const { return RedProgress + RedDetection + RedTracking + RedTime + RedTerminal; }
};

/** Observable tracking surrogate. A lost detection clears the current track. */
USTRUCT(BlueprintType)
struct TRIADSENSORFUSION_API FTRIADRLTrackState
{
    GENERATED_BODY()
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") bool bCurrentlyDetected = false;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") bool bTracked = false;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") bool bEverDetected = false;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") int32 ConsecutiveDetectionSteps = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") int32 FirstDetectionStep = INDEX_NONE;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") int32 LastDetectionStep = INDEX_NONE;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") int32 DetectedStepCount = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") int32 TrackedStepCount = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL") int32 LastUpdateStep = 0;
};

USTRUCT(BlueprintType)
struct TRIADSENSORFUSION_API FTRIADRLSensorCandidate
{
    GENERATED_BODY()

    /** Approved simulation option. Several orientations may share one MountId. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Catalogue")
    FString CandidateId;
    /** Stable sensor identity used by variable-catalogue policies and checkpoint manifests. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Catalogue")
    FString SensorProfileId = TEXT("default");
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Catalogue")
    FString MountId;
    /** If true, the action supplies normalized East/North coordinates inside the placement annulus. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Catalogue")
    bool bAllowDynamicPosition = false;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Catalogue")
    FVector OffsetEnuMeters = FVector::ZeroVector;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Catalogue")
    double YawDegrees = 0.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Catalogue")
    double PitchDegrees = 0.0;
    /** Policy-visible analytic coverage envelope; separate from render-backed PTZ capture optics. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Catalogue")
    double HorizontalFovDegrees = 90.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Catalogue")
    double VerticalFovDegrees = 90.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Catalogue")
    double CostUnits = 1.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Catalogue")
    bool bApproved = true;
    /** Require collision-supported ground, unless this is an explicit authored mount. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Catalogue")
    bool bProjectToGround = true;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Catalogue")
    FTRIADGeodeticSensorNode Sensor;
};

USTRUCT(BlueprintType)
struct TRIADSENSORFUSION_API FTRIADRLTrainingConfig
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL")
    FString SchemaVersion = TEXT("triad.rl_training.v4");

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL")
    FString ScenarioId = TEXT("unconfigured");
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL")
    int32 Difficulty = 0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Blue")
    TArray<FTRIADRLSensorCandidate> SensorCandidates;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL")
    bool bEnabled = false;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL")
    int32 RandomSeed = 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL", meta = (ClampMin = "0.02", ClampMax = "2.0"))
    double FixedStepSeconds = 0.25;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL", meta = (ClampMin = "1", ClampMax = "100000"))
    int32 EpisodeHorizonSteps = 600;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL", meta = (ClampMin = "1", ClampMax = "32"))
    int32 MaximumSensorSites = 6;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL", meta = (ClampMin = "0.0"))
    double SensorBudgetUnits = 12.0;

    /** Validity boundary for approved mounts centred on the map-authored objective. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Blue", meta = (ClampMin = "50.0"))
    double BluePlacementRadiusMeters = 1000.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Blue", meta = (ClampMin = "0.0"))
    double BlueMinimumObjectiveStandoffMeters = 100.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Blue", meta = (ClampMin = "0.0"))
    double MinimumSensorSeparationMeters = 25.0;

    /** Continuous Red deployment annulus. It may extend beyond visible authored geometry but remains bounded. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red", meta = (ClampMin = "10.0"))
    double RedMinimumSpawnRadiusMeters = 1200.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red", meta = (ClampMin = "10.0"))
    double RedMaximumSpawnRadiusMeters = 2000.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red")
    double RedMinimumBearingDegrees = -180.0;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red")
    double RedMaximumBearingDegrees = 180.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red", meta = (ClampMin = "5.0"))
    double RedMinimumAltitudeMeters = 60.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red", meta = (ClampMin = "5.0"))
    double RedMaximumAltitudeMeters = 180.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red", meta = (ClampMin = "1", ClampMax = "64"))
    int32 RedMinimumSwarmSize = 1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red", meta = (ClampMin = "1", ClampMax = "64"))
    int32 RedMaximumSwarmSize = 8;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red", meta = (ClampMin = "1.0", ClampMax = "100.0"))
    double RedMinimumFormationSpacingMeters = 10.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red", meta = (ClampMin = "1.0", ClampMax = "100.0"))
    double RedMaximumFormationSpacingMeters = 40.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL", meta = (ClampMin = "0.1", ClampMax = "50.0"))
    double MaximumTargetSpeedMetersPerSecond = 15.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL", meta = (ClampMin = "100.0"))
    double MaximumDistanceFromZoneMeters = 2000.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL", meta = (ClampMin = "1", ClampMax = "32"))
    int32 ConfirmationNodeCount = 2;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL")
    int32 TrackConfirmationSteps = 3;

    /** Consecutive all-target tracked steps count as abstract defence, not physical interception. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL")
    int32 DefenceTrackHoldSteps = 4;

    /** Curriculum control. Zero axes are inactive in the policy and native dynamics. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red")
    FVector RedMovementAxisMask = FVector(1.0, 1.0, 1.0);

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL")
    FTRIADRLProtectedZoneDefinition ProtectedZone;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL")
    FTRIADRLRewardWeights Rewards;
};

UCLASS(BlueprintType)
class TRIADSENSORFUSION_API UTRIADRLTrainingDefinition : public UPrimaryDataAsset
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    FTRIADRLTrainingConfig Config;
};

USTRUCT(BlueprintType)
struct TRIADSENSORFUSION_API FTRIADRLBlueAction
{
    GENERATED_BODY()

    /** Index into the server-approved catalogue; -1 is never a valid placement. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Blue")
    int32 CatalogueIndex = INDEX_NONE;

    /** Continuous local East/North coordinates in the configured placement annulus.
     *  Ignored by fixed-mount catalogue entries and by the stop action.
     */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Blue")
    FVector2D NormalizedPosition = FVector2D::ZeroVector;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Blue")
    bool bStopPlacement = false;
};

USTRUCT(BlueprintType)
struct TRIADSENSORFUSION_API FTRIADRLRedDeploymentAction
{
    GENERATED_BODY()

    /** All values are normalized to [-1, 1] and mapped into the configured bounded domain. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red")
    double NormalizedBearing = 0.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red")
    double NormalizedRadius = 0.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red")
    double NormalizedAltitude = 0.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red")
    double NormalizedSwarmSize = 0.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red")
    double NormalizedFormationSpacing = 0.0;
};

USTRUCT(BlueprintType)
struct TRIADSENSORFUSION_API FTRIADRLRedAction
{
    GENERATED_BODY()

    /** Each component is normalized to [-1, 1] and is never a motor or vehicle command. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TRIAD|RL|Red")
    FVector NormalizedVelocityEnu = FVector::ZeroVector;
};

USTRUCT(BlueprintType)
struct TRIADSENSORFUSION_API FTRIADRLTargetObservation
{
    GENERATED_BODY()

    /** Target displacement from the protected-zone centre, normalized by the configured AOI radius. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL|Observation")
    FVector NormalizedZoneRelativeEnu = FVector::ZeroVector;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL|Observation")
    FVector NormalizedVelocityEnu = FVector::ZeroVector;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL|Observation")
    bool bConfirmedDetected = false;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL|Observation")
    bool bCurrentlyDetected = false;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL|Observation")
    bool bTracked = false;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL|Observation")
    int32 ConsecutiveDetectionSteps = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL|Observation")
    int32 FirstDetectionStep = INDEX_NONE;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL|Observation")
    int32 LastDetectionStep = INDEX_NONE;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL|Observation")
    int32 TrackedStepCount = 0;
};

USTRUCT(BlueprintType)
struct TRIADSENSORFUSION_API FTRIADRLMountObservation
{
    GENERATED_BODY()
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    int32 CatalogueIndex = INDEX_NONE;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    FString CandidateId;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    FString SensorProfileId;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    FString MountId;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    FVector OffsetEnuMeters = FVector::ZeroVector;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    FVector SurfaceNormal = FVector::UpVector;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    bool bSurfaceResolved = false;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    bool bOccupied = false;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    bool bEnabled = false;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    double YawDegrees = 0.0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    double PitchDegrees = 0.0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    double HorizontalFovDegrees = 0.0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    double VerticalFovDegrees = 0.0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    double RangeMeters = 0.0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    double CostUnits = 0.0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    int32 SensorModalityMask = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    bool bAllowDynamicPosition = false;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    int32 CurrentlyDetectingTargetCount = 0;
};

USTRUCT(BlueprintType)
struct TRIADSENSORFUSION_API FTRIADRLStepResult
{
    GENERATED_BODY()

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    FString SchemaVersion = TEXT("triad.rl_step.v4");
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    FString ScenarioId;
    /** MD5 is used only as a fast fail-closed experiment identity, not for security. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    FString ConfigFingerprint;
    /** Monotonically increases per accepted episode transition, not wall frame. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    int32 TransitionIndex = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    int32 EpisodeSeed = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    double RemainingBudgetUnits = 0.0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    int32 RemainingSiteCount = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    TArray<FTRIADRLMountObservation> MountObservations;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    TArray<int32> PlacedCatalogueIndices;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    ETRIADRLPhase Phase = ETRIADRLPhase::Inactive;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    ETRIADRLTerminationReason TerminationReason = ETRIADRLTerminationReason::None;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    int32 StepIndex = 0;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    int32 ActiveTargetCount = 0;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    int32 DetectedTargetCount = 0;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    int32 CurrentlyDetectedTargetCount = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    int32 TrackedTargetCount = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    int32 AllTargetsTrackedSteps = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    FTRIADRLRewardBreakdown StepRewards;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    int32 InvalidActionCount = 0;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    FString LastBlueAction;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    FString LastRedDeploymentAction;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    TArray<FVector> LastRedMovementActions;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    double MinimumZoneDistanceMeters = 0.0;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    double BlueReward = 0.0;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    double RedReward = 0.0;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    bool bTerminal = false;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    TArray<bool> BlueActionMask;

    /** Committed Blue sites in objective-relative EN coordinates normalized by BluePlacementRadiusMeters. */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    TArray<FVector2D> SensorObservations;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "TRIAD|RL")
    TArray<FTRIADRLTargetObservation> TargetObservations;
};
