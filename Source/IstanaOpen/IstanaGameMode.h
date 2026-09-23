#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "GameFramework/HUD.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/PlayerController.h"
#include "IstanaGameMode.generated.h"

class UCameraComponent;
class USphereComponent;
class UInstancedStaticMeshComponent;
class ABlueTeamCoordinator;
class ARedTeamManager;

/** A persistent native owner for authored planting instances in saved/cooked maps. */
UCLASS()
class ISTANAOPEN_API AIstanaPlantingGroup : public AActor
{
    GENERATED_BODY()
public:
    AIstanaPlantingGroup();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Planting")
    TObjectPtr<UInstancedStaticMeshComponent> Instances;
};

/** Standalone local viewer. No sensor runtime, provider token, or network is required. */
UCLASS()
class ISTANAOPEN_API AIstanaGameMode : public AGameModeBase
{
    GENERATED_BODY()
public:
    AIstanaGameMode();
    virtual void StartPlay() override;
    // Applies only opt-in Blue-live scenario overrides. Kept separate from the
    // saved map defaults so native automation can verify the mode boundary.
    static void ConfigureBlueLiveApproach(ARedTeamManager& Manager, ABlueTeamCoordinator& Coordinator,
        bool bWarningApproachV2, bool bDelayedDetectionDemo, bool bTrainingWorkbench = false);
};

UCLASS()
class ISTANAOPEN_API AIstanaCameraPawn : public APawn
{
    GENERATED_BODY()
public:
    AIstanaCameraPawn();
    virtual void Tick(float DeltaSeconds) override;
    void SetView(int32 Index);
    void ToggleTour();
    bool IsTouring() const { return bTouring; }
    int32 GetCurrentView() const { return CurrentView; }
    FVector GetViewLocation() const;
    FRotator GetViewRotation() const;

protected:
    virtual void BeginPlay() override;

private:
    UPROPERTY() TObjectPtr<USphereComponent> Body;
    UPROPERTY() TObjectPtr<UCameraComponent> Camera;
    FVector Velocity = FVector::ZeroVector;
    bool bTouring = false;
    float TourAngle = -HALF_PI;
    int32 CurrentView = 1;
};

UCLASS()
class ISTANAOPEN_API AIstanaPlayerController : public APlayerController
{
    GENERATED_BODY()
public:
    AIstanaPlayerController();
    virtual void Tick(float DeltaSeconds) override;
    bool IsMouseReleased() const { return bMouseReleased; }
    bool IsHelpVisible() const { return bHelp; }
    bool IsPhotoMode() const { return bPhoto; }
    bool IsHighQuality() const { return bHighQuality; }
    float GetSmoothedFPS() const { return SmoothedFPS; }

    UFUNCTION(Exec) void IstanaView(int32 Index = 1);
    UFUNCTION(Exec) void IstanaTour();
    UFUNCTION(Exec) void IstanaQuality();

protected:
    virtual void BeginPlay() override;
    virtual void SetupInputComponent() override;

private:
    void ViewOne();
    void ViewTwo();
    void ViewThree();
    void ViewFour();
    void ToggleHelp();
    void TogglePhoto();
    void Escape();
    void Resume();
    void SetMouseReleased(bool bReleased);
    void ApplyQuality();
    void ApplyRequestedResolution();
    void WriteRunReport();

    bool bMouseReleased = false;
    bool bHelp = false;
    bool bPhoto = false;
    bool bHighQuality = false;
    bool bCaptureRequested = false;
    bool bRunFinished = false;
    bool bAutoQuit = false;
    bool bScreenshotSaved = false;
    bool bCaptureDestinationReady = false;
    FString CapturePath;
    FString ReportPath;
    float CaptureDelay = 20.f;
    double RunStarted = 0;
    double CaptureStarted = 0;
    double SampleTime = 0;
    double SampleSeconds = 0;
    int32 SampleFrames = 0;
    int32 RequestedWidth = 0;
    int32 RequestedHeight = 0;
    float SmoothedFPS = 60.f;
};

UCLASS()
class ISTANAOPEN_API AIstanaHUD : public AHUD
{
    GENERATED_BODY()
public:
    virtual void DrawHUD() override;
};
