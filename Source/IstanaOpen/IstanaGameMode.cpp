#include "IstanaGameMode.h"

#include "Camera/CameraComponent.h"
#include "Components/InputComponent.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Components/SphereComponent.h"
#include "Engine/Canvas.h"
#include "Engine/Engine.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/TargetPoint.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/GameUserSettings.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformTime.h"
#include "InputCoreTypes.h"
#include "Kismet/KismetSystemLibrary.h"
#include "Misc/CommandLine.h"
#include "Misc/FileHelper.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "RHI.h"
#include "Scalability.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "UnrealClient.h"
#include "Simulation/BlueTeam/BlueTeamCoordinator.h"
#include "Simulation/RedTeam/RedTeamAgentBridge.h"
#include "Simulation/RedTeam/RedTeamManager.h"

AIstanaPlantingGroup::AIstanaPlantingGroup()
{
    PrimaryActorTick.bCanEverTick = false;
    Instances = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("Instances"));
    SetRootComponent(Instances);
    Instances->SetMobility(EComponentMobility::Static);
    Instances->SetCanEverAffectNavigation(false);
    Instances->SetCollisionEnabled(ECollisionEnabled::NoCollision);
}

AIstanaGameMode::AIstanaGameMode()
{
    DefaultPawnClass = AIstanaCameraPawn::StaticClass();
    PlayerControllerClass = AIstanaPlayerController::StaticClass();
    HUDClass = AIstanaHUD::StaticClass();
}

void AIstanaGameMode::StartPlay()
{
    ARedTeamAgentBridge* PendingLiveBridge = nullptr;
    // Configure actors before their BeginPlay so the external episode owns the clock.
    // This is an opt-in runtime setup; the saved landscape stays usable as a viewer.
    if (FParse::Param(FCommandLine::Get(), TEXT("IstanaBlueLive")))
    {
        TArray<ARedTeamManager*> Managers;
        for (TActorIterator<ARedTeamManager> It(GetWorld()); It; ++It) Managers.Add(*It);
        int32 Port = 8765;
        FParse::Value(FCommandLine::Get(), TEXT("IstanaBluePort="), Port);
        if (Managers.Num() != 1 || !IsValid(Managers[0]->ObjectiveTarget) || Port < 1024 || Port > 65535
            || FParse::Param(FCommandLine::Get(), TEXT("IstanaSwarmProfile")))
        {
            UE_LOG(LogTemp, Error, TEXT("IstanaBlueLive requires exactly one RedTeamManager with an objective, a valid port, and no swarm profiling override."));
            // A rejected setup must not silently run a seeded scenario.
            for (auto* Manager : Managers) { Manager->bAutoInitialize = false; Manager->bAutoAdvance = false; }
        }
        else
        {
            auto* Manager = Managers[0];
            Manager->PlacementSource = ERedTeamPlacementSource::AgentPlacement;
            Manager->bAutoInitialize = false;
            Manager->bAutoAdvance = false;
            Manager->bEnableDemoKeyboard = false;
            auto* Coordinator = Manager->BlueCoordinator.Get();
            if (!IsValid(Coordinator)) Coordinator = GetWorld()->SpawnActor<ABlueTeamCoordinator>();
            if (IsValid(Coordinator))
            {
                Coordinator->Manager = Manager;
                Manager->BlueCoordinator = Coordinator;
                ARedTeamAgentBridge* Bridge = nullptr;
                for (TActorIterator<ARedTeamAgentBridge> It(GetWorld()); It; ++It)
                    if (It->Manager == Manager) { Bridge = *It; break; }
                if (!Bridge) Bridge = GetWorld()->SpawnActor<ARedTeamAgentBridge>();
                if (Bridge)
                {
                    Bridge->Manager = Manager;
                    Bridge->Port = Port;
                    Bridge->IdleTimeoutSeconds = 300;
                    Bridge->bStartOnBeginPlay = true;
                    PendingLiveBridge = Bridge;
                }
                else UE_LOG(LogTemp, Error, TEXT("IstanaBlueLive could not create the loopback bridge."));
            }
            else UE_LOG(LogTemp, Error, TEXT("IstanaBlueLive could not create the Blue coordinator."));
        }
    }
    Super::StartPlay();
    if (PendingLiveBridge)
    {
        FString BridgeError;
        if (PendingLiveBridge->StartBridge(BridgeError))
        {
            UE_LOG(LogTemp, Display, TEXT("IstanaBlueLive listening on 127.0.0.1:%d; awaiting Python planner."), PendingLiveBridge->Port);
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("IstanaBlueLive could not start the loopback bridge: %s"), *BridgeError);
        }
    }
}

AIstanaCameraPawn::AIstanaCameraPawn()
{
    PrimaryActorTick.bCanEverTick = true;
    Body = CreateDefaultSubobject<USphereComponent>(TEXT("CameraBody"));
    SetRootComponent(Body);
    Body->InitSphereRadius(28.f);
    Body->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    Body->SetCanEverAffectNavigation(false);
    Camera = CreateDefaultSubobject<UCameraComponent>(TEXT("Camera"));
    Camera->SetupAttachment(Body);
    Camera->FieldOfView = 65.f;
    Camera->bUsePawnControlRotation = false;
    Camera->PostProcessSettings.bOverride_MotionBlurAmount = true;
    Camera->PostProcessSettings.MotionBlurAmount = 0.f;
    Camera->PostProcessSettings.bOverride_VignetteIntensity = true;
    Camera->PostProcessSettings.VignetteIntensity = 0.15f;
    Camera->PostProcessBlendWeight = 1.f;
}

void AIstanaCameraPawn::BeginPlay()
{
    Super::BeginPlay();
    int32 InitialView = 1;
    FParse::Value(FCommandLine::Get(), TEXT("IstanaCaptureView="), InitialView);
    SetView(InitialView);
}

void AIstanaCameraPawn::SetView(int32 Index)
{
    CurrentView = FMath::Clamp(Index, 1, 4);
    FVector Position;
    FVector Target;
    switch (CurrentView)
    {
    case 2: Position = FVector(-7000, -9500, 700); Target = FVector(0, -1000, 1100); break;
    case 3: Position = FVector(10000, -25000, 27000); Target = FVector(0, 0, 800); break;
    case 4: Position = FVector(16000, -43000, 1800); Target = FVector(0, -15000, 1400); break;
    default: Position = FVector(7000, -13500, 3300); Target = FVector(0, -500, 1250); break;
    }
    bTouring = false;
    Velocity = FVector::ZeroVector;
    SetActorLocationAndRotation(Position, (Target - Position).Rotation());
}

void AIstanaCameraPawn::ToggleTour()
{
    bTouring = !bTouring;
    TourAngle = FMath::Atan2(GetActorLocation().Y, GetActorLocation().X);
    Velocity = FVector::ZeroVector;
}

FVector AIstanaCameraPawn::GetViewLocation() const { return Camera->GetComponentLocation(); }
FRotator AIstanaCameraPawn::GetViewRotation() const { return Camera->GetComponentRotation(); }

void AIstanaCameraPawn::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);
    AIstanaPlayerController* PC = Cast<AIstanaPlayerController>(GetController());
    if (!PC || !PC->IsLocalController() || PC->IsMouseReleased()) return;
    const float Dt = FMath::Min(DeltaSeconds, 0.1f);
    const auto Axis = [PC](const FKey& Positive, const FKey& Negative)
    { return float(PC->IsInputKeyDown(Positive)) - float(PC->IsInputKeyDown(Negative)); };
    FVector Input(Axis(EKeys::W, EKeys::S), Axis(EKeys::D, EKeys::A), Axis(EKeys::E, EKeys::Q));
    float MouseX = 0, MouseY = 0;
    PC->GetInputMouseDelta(MouseX, MouseY);
    if (!Input.IsNearlyZero() || FMath::Abs(MouseX) + FMath::Abs(MouseY) > 0.2f) bTouring = false;
    if (bTouring)
    {
        TourAngle += Dt * 0.075f;
        const FVector Position(FMath::Cos(TourAngle) * 20500.f, FMath::Sin(TourAngle) * 20500.f,
            6300.f + 1400.f * FMath::Sin(TourAngle * 0.65f));
        const FVector SmoothPosition = FMath::VInterpTo(GetActorLocation(), Position, Dt, 0.85f);
        const FRotator SmoothRotation = FMath::RInterpTo(GetActorRotation(),
            (FVector(0, 0, 2000) - SmoothPosition).Rotation(), Dt, 0.85f);
        SetActorLocationAndRotation(SmoothPosition, SmoothRotation);
        return;
    }
    FRotator Look = GetActorRotation();
    Look.Yaw += MouseX * 0.12f;
    Look.Pitch = FMath::Clamp(Look.Pitch - MouseY * 0.12f, -88.f, 88.f);
    Look.Roll = 0;
    SetActorRotation(Look);
    const FVector Direction = (GetActorForwardVector() * Input.X + GetActorRightVector() * Input.Y
        + FVector::UpVector * Input.Z).GetClampedToMaxSize(1.f);
    float Speed = 2200.f;
    if (PC->IsInputKeyDown(EKeys::LeftShift) || PC->IsInputKeyDown(EKeys::RightShift)) Speed *= 3.5f;
    if (PC->IsInputKeyDown(EKeys::LeftControl) || PC->IsInputKeyDown(EKeys::RightControl)) Speed *= 0.15f;
    Velocity = FMath::VInterpTo(Velocity, Direction * Speed, Dt, 6.f);
    FVector Next = GetActorLocation() + Velocity * Dt;
    Next.X = FMath::Clamp(Next.X, -85000., 85000.);
    Next.Y = FMath::Clamp(Next.Y, -85000., 85000.);
    Next.Z = FMath::Clamp(Next.Z, 150., 65000.);
    SetActorLocation(Next);
}

AIstanaPlayerController::AIstanaPlayerController()
{
    PrimaryActorTick.bCanEverTick = true;
    bShowMouseCursor = false;
}

void AIstanaPlayerController::BeginPlay()
{
    Super::BeginPlay();
    SetMouseReleased(false);
    RunStarted = FPlatformTime::Seconds();
    SampleTime = RunStarted;
    FParse::Value(FCommandLine::Get(), TEXT("IstanaCapture="), CapturePath);
    FParse::Value(FCommandLine::Get(), TEXT("IstanaReport="), ReportPath);
    FParse::Value(FCommandLine::Get(), TEXT("IstanaCaptureDelay="), CaptureDelay);
    CaptureDelay = FMath::Max(5.f, CaptureDelay);
    bAutoQuit = FParse::Param(FCommandLine::Get(), TEXT("IstanaAutoQuit"));
    bPhoto = FParse::Param(FCommandLine::Get(), TEXT("IstanaPhoto"));
    bHighQuality = FParse::Param(FCommandLine::Get(), TEXT("IstanaHigh"))
        && !FParse::Param(FCommandLine::Get(), TEXT("IstanaMedium"));
    ApplyQuality();
    ApplyRequestedResolution();
}

void AIstanaPlayerController::SetupInputComponent()
{
    Super::SetupInputComponent();
    InputComponent->BindKey(EKeys::One, IE_Pressed, this, &AIstanaPlayerController::ViewOne);
    InputComponent->BindKey(EKeys::Two, IE_Pressed, this, &AIstanaPlayerController::ViewTwo);
    InputComponent->BindKey(EKeys::Three, IE_Pressed, this, &AIstanaPlayerController::ViewThree);
    InputComponent->BindKey(EKeys::Four, IE_Pressed, this, &AIstanaPlayerController::ViewFour);
    InputComponent->BindKey(EKeys::T, IE_Pressed, this, &AIstanaPlayerController::IstanaTour);
    InputComponent->BindKey(EKeys::F, IE_Pressed, this, &AIstanaPlayerController::IstanaQuality);
    InputComponent->BindKey(EKeys::F1, IE_Pressed, this, &AIstanaPlayerController::ToggleHelp);
    InputComponent->BindKey(EKeys::P, IE_Pressed, this, &AIstanaPlayerController::TogglePhoto);
    InputComponent->BindKey(EKeys::Escape, IE_Pressed, this, &AIstanaPlayerController::Escape);
    InputComponent->BindKey(EKeys::Enter, IE_Pressed, this, &AIstanaPlayerController::Resume);
    InputComponent->BindKey(EKeys::LeftMouseButton, IE_Pressed, this, &AIstanaPlayerController::Resume);
}

void AIstanaPlayerController::ViewOne() { IstanaView(1); }
void AIstanaPlayerController::ViewTwo() { IstanaView(2); }
void AIstanaPlayerController::ViewThree() { IstanaView(3); }
void AIstanaPlayerController::ViewFour() { IstanaView(4); }
void AIstanaPlayerController::ToggleHelp() { bHelp = !bHelp; }
void AIstanaPlayerController::TogglePhoto() { bPhoto = !bPhoto; }
void AIstanaPlayerController::IstanaView(int32 Index)
{
    if (AIstanaCameraPawn* Camera = Cast<AIstanaCameraPawn>(GetPawn())) Camera->SetView(Index);
    Resume();
}
void AIstanaPlayerController::IstanaTour()
{
    if (AIstanaCameraPawn* Camera = Cast<AIstanaCameraPawn>(GetPawn())) Camera->ToggleTour();
    Resume();
}
void AIstanaPlayerController::IstanaQuality() { bHighQuality = !bHighQuality; ApplyQuality(); }
void AIstanaPlayerController::ApplyQuality()
{
    Scalability::FQualityLevels Quality;
    Quality.SetFromSingleQualityLevel(bHighQuality ? 2 : 1);
    Quality.ResolutionQuality = bHighQuality ? 100.f : 85.f;
    Scalability::SetQualityLevels(Quality);
}
void AIstanaPlayerController::ApplyRequestedResolution()
{
    // UE normally clamps a window to the host desktop/work area. Explicit capture
    // dimensions must also work on small remote or offscreen desktops.
    const bool HasWidth = FParse::Value(FCommandLine::Get(), TEXT("ResX="), RequestedWidth);
    const bool HasHeight = FParse::Value(FCommandLine::Get(), TEXT("ResY="), RequestedHeight);
    if (!HasWidth && !HasHeight) return;
    if (!HasWidth) RequestedWidth = FMath::RoundToInt(RequestedHeight * (16.f / 9.f));
    if (!HasHeight) RequestedHeight = FMath::RoundToInt(RequestedWidth * (9.f / 16.f));
    if (RequestedWidth < 320 || RequestedHeight < 180 || RequestedWidth > 16384 || RequestedHeight > 16384)
    {
        UE_LOG(LogTemp, Warning, TEXT("Ignoring invalid requested viewport size %dx%d"), RequestedWidth, RequestedHeight);
        RequestedWidth = RequestedHeight = 0;
        return;
    }
    EWindowMode::Type WindowMode = EWindowMode::Windowed;
    if (GEngine && GEngine->GetGameUserSettings()) WindowMode = GEngine->GetGameUserSettings()->GetFullscreenMode();
    if (FParse::Param(FCommandLine::Get(), TEXT("windowed"))) WindowMode = EWindowMode::Windowed;
    if (FParse::Param(FCommandLine::Get(), TEXT("fullscreen"))) WindowMode = EWindowMode::Fullscreen;
    UGameUserSettings::RequestResolutionChange(RequestedWidth, RequestedHeight, WindowMode, false);
}
void AIstanaPlayerController::Escape()
{
    if (bMouseReleased)
    {
        UKismetSystemLibrary::QuitGame(this, this, EQuitPreference::Quit, false);
        return;
    }
    SetMouseReleased(true);
}
void AIstanaPlayerController::Resume() { if (bMouseReleased) SetMouseReleased(false); }
void AIstanaPlayerController::SetMouseReleased(bool bReleased)
{
    bMouseReleased = bReleased;
    bShowMouseCursor = bReleased;
    if (bReleased)
    {
        FInputModeGameAndUI Input;
        Input.SetLockMouseToViewportBehavior(EMouseLockMode::DoNotLock);
        Input.SetHideCursorDuringCapture(false);
        SetInputMode(Input);
    }
    else
    {
        FInputModeGameOnly Input;
        Input.SetConsumeCaptureMouseDown(false);
        SetInputMode(Input);
    }
}

void AIstanaPlayerController::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);
    const double Now = FPlatformTime::Seconds();
    const double FrameSeconds = Now - SampleTime;
    SampleTime = Now;
    if (FrameSeconds > 0) SmoothedFPS = FMath::Lerp(SmoothedFPS, float(1.0 / FrameSeconds), 0.04f);
    if (Now - RunStarted > 5 && FrameSeconds > 0)
    {
        SampleSeconds += FrameSeconds;
        ++SampleFrames;
    }
    if (bRunFinished || (CapturePath.IsEmpty() && ReportPath.IsEmpty() && !bAutoQuit)) return;
    if (!bCaptureRequested && Now - RunStarted >= CaptureDelay)
    {
        bCaptureRequested = true;
        CaptureStarted = Now;
        if (!CapturePath.IsEmpty())
        {
            IFileManager::Get().MakeDirectory(*FPaths::GetPath(CapturePath), true);
            bCaptureDestinationReady = !IFileManager::Get().FileExists(*CapturePath)
                || IFileManager::Get().Delete(*CapturePath, true, false, true);
            if (bCaptureDestinationReady)
                FScreenshotRequest::RequestScreenshot(CapturePath, true, false, false);
        }
    }
    if (bCaptureRequested && Now - CaptureStarted > 3)
    {
        bScreenshotSaved = CapturePath.IsEmpty() || (bCaptureDestinationReady
            && IFileManager::Get().FileSize(*CapturePath) > 0);
        if (!bScreenshotSaved && Now - CaptureStarted < 30) return;
        WriteRunReport();
        bRunFinished = true;
        if (bAutoQuit) UKismetSystemLibrary::QuitGame(this, this, EQuitPreference::Quit, false);
    }
}

void AIstanaPlayerController::WriteRunReport()
{
    if (ReportPath.IsEmpty()) return;
    const AIstanaCameraPawn* Camera = Cast<AIstanaCameraPawn>(GetPawn());
    TSharedRef<FJsonObject> Report = MakeShared<FJsonObject>();
    Report->SetStringField(TEXT("project"), TEXT("IstanaOpen"));
    Report->SetStringField(TEXT("map"), GetWorld()->GetMapName());
    Report->SetBoolField(TEXT("standalone_world"), GetWorld()->WorldType == EWorldType::Game);
    Report->SetBoolField(TEXT("camera_possessed"), Camera != nullptr);
    Report->SetBoolField(TEXT("screenshot_saved"), bScreenshotSaved);
    Report->SetStringField(TEXT("screenshot"), CapturePath);
    Report->SetStringField(TEXT("gpu"), GRHIAdapterName);
    Report->SetStringField(TEXT("quality"), bHighQuality ? TEXT("High") : TEXT("Medium"));
    Report->SetNumberField(TEXT("average_fps_after_warmup"), SampleSeconds > 0 ? SampleFrames / SampleSeconds : 0);
    Report->SetNumberField(TEXT("measured_frames"), SampleFrames);
    Report->SetNumberField(TEXT("measured_seconds"), SampleSeconds);
    Report->SetNumberField(TEXT("elapsed_seconds"), FPlatformTime::Seconds() - RunStarted);
    int32 ViewportWidth = 0, ViewportHeight = 0;
    GetViewportSize(ViewportWidth, ViewportHeight);
    Report->SetNumberField(TEXT("viewport_width"), ViewportWidth);
    Report->SetNumberField(TEXT("viewport_height"), ViewportHeight);
    if (RequestedWidth > 0 && RequestedHeight > 0)
    {
        Report->SetNumberField(TEXT("requested_viewport_width"), RequestedWidth);
        Report->SetNumberField(TEXT("requested_viewport_height"), RequestedHeight);
        Report->SetBoolField(TEXT("requested_resolution_honored"), ViewportWidth == RequestedWidth && ViewportHeight == RequestedHeight);
    }
    int32 StaticMeshActors = 0;
    for (TActorIterator<AStaticMeshActor> It(GetWorld()); It; ++It) ++StaticMeshActors;
    Report->SetNumberField(TEXT("static_mesh_actors"), StaticMeshActors);
    if (Camera)
    {
        Report->SetStringField(TEXT("camera_location_cm"), Camera->GetViewLocation().ToString());
        Report->SetStringField(TEXT("camera_rotation_degrees"), Camera->GetViewRotation().ToString());
        Report->SetNumberField(TEXT("view"), Camera->GetCurrentView());
    }
    FString Json;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Json);
    FJsonSerializer::Serialize(Report, Writer);
    IFileManager::Get().MakeDirectory(*FPaths::GetPath(ReportPath), true);
    FFileHelper::SaveStringToFile(Json, *ReportPath);
}

void AIstanaHUD::DrawHUD()
{
    Super::DrawHUD();
    if (!Canvas || !GEngine) return;
    const AIstanaPlayerController* PC = Cast<AIstanaPlayerController>(PlayerOwner);
    if (!PC || (PC->IsPhotoMode() && !PC->IsMouseReleased())) return;
    const float Width = Canvas->SizeX;
    const float Height = Canvas->SizeY;
    const float S = FMath::Clamp(Width / 1600.f, 0.72f, 1.5f);
    const float Margin = 30.f * S;
    const FLinearColor Ink(0.025f, 0.055f, 0.065f, 0.86f);
    const FLinearColor White(0.94f, 0.95f, 0.90f, 1.f);
    const FLinearColor Gold(0.83f, 0.68f, 0.39f, 1.f);
    const FLinearColor Muted(0.68f, 0.77f, 0.76f, 1.f);
    UFont* Font = GEngine->GetSmallFont();
    DrawRect(Ink, Margin, Margin, 355 * S, 94 * S);
    DrawRect(Gold, Margin, Margin, 3 * S, 94 * S);
    DrawText(TEXT("ISTANA  /  SINGAPORE"), White, Margin + 18 * S, Margin + 13 * S, GEngine->GetMediumFont(), 1.05f * S);
    DrawText(TEXT("An open, offline architectural landscape"), Muted, Margin + 18 * S, Margin + 44 * S, Font, S);
    DrawText(TEXT("PUBLIC EXTERIOR RECONSTRUCTION"), Gold, Margin + 18 * S, Margin + 66 * S, Font, 0.87f * S);
    const FString Status = FString::Printf(TEXT("%s  |  %.0f FPS"), PC->IsHighQuality() ? TEXT("HIGH") : TEXT("MEDIUM"), PC->GetSmoothedFPS());
    DrawRect(Ink, Width - Margin - 180 * S, Margin, 180 * S, 34 * S);
    DrawText(Status, White, Width - Margin - 168 * S, Margin + 9 * S, Font, S);
    const float FooterHeight = 70 * S;
    DrawRect(Ink, 0, Height - FooterHeight, Width, FooterHeight);
    DrawText(TEXT("1  PALACE     2  GARDEN     3  AERIAL     4  APPROACH     T  TOUR"), White,
        Margin, Height - FooterHeight + 11 * S, Font, S);
    DrawText(TEXT("WASD + mouse explore   |   E/Q height   |   F quality   |   P photo   |   F1 controls"), Muted,
        Margin, Height - FooterHeight + 34 * S, Font, 0.94f * S);
    // Use the full attribution in narrow windows too, without colliding with controls.
    if (Width > 1100)
    {
        DrawText(TEXT("Map data: \u00A9 OpenStreetMap contributors / ODbL"), Muted,
            Width - Margin - 325 * S, Height - FooterHeight + 13 * S, Font, 0.83f * S);
        DrawText(TEXT("Visual approximation. Exterior only."), Muted,
            Width - Margin - 325 * S, Height - FooterHeight + 34 * S, Font, 0.83f * S);
    }
    else
    {
        DrawText(TEXT("Map data: \u00A9 OpenStreetMap contributors / ODbL"), Muted,
            Margin, Height - FooterHeight - 21 * S, Font, 0.86f * S);
    }
    if (PC->IsHelpVisible() || PC->IsMouseReleased())
    {
        const float PanelX = Width * 0.5f - 250 * S;
        const float PanelY = Height * 0.5f - 180 * S;
        DrawRect(Ink, PanelX, PanelY, 500 * S, 334 * S);
        DrawRect(Gold, PanelX, PanelY, 500 * S, 2 * S);
        DrawText(PC->IsMouseReleased() ? TEXT("EXPLORATION PAUSED") : TEXT("EXPLORE THE LANDSCAPE"), White,
            PanelX + 26 * S, PanelY + 25 * S, GEngine->GetMediumFont(), 1.1f * S);
        const TCHAR* Lines[] = {
            TEXT("W / A / S / D          Fly and strafe"),
            TEXT("Mouse                  Look around"),
            TEXT("E / Q                    Rise / descend"),
            TEXT("Shift / Ctrl             Fast / precise movement"),
            TEXT("1 / 2 / 3 / 4          Curated viewpoints"),
            TEXT("T / F / P                Tour / quality / photo mode"),
            TEXT("F1                         Show or hide these controls"),
            TEXT("Esc                        Release mouse, then Esc to quit")
        };
        for (int32 I = 0; I < UE_ARRAY_COUNT(Lines); ++I)
            DrawText(Lines[I], Muted, PanelX + 26 * S, PanelY + (70 + I * 24) * S, Font, S);
        if (PC->IsMouseReleased()) DrawText(TEXT("CLICK OR PRESS ENTER TO CONTINUE"), Gold,
            PanelX + 26 * S, PanelY + 291 * S, Font, S);
    }
}
