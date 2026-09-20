#include "Simulation/RedTeam/RedTeamAgentBridge.h"
#include "Simulation/RedTeam/RedTeamManager.h"
#include "Simulation/BlueTeam/BlueTeamCoordinator.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "IPAddress.h"
#include "JsonObjectConverter.h"
#include "Serialization/JsonSerializer.h"
#include "Policies/CondensedJsonPrintPolicy.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformTime.h"
#include "Engine/Engine.h"
#include "Engine/GameViewportClient.h"
#include "Engine/TargetPoint.h"
#include "EngineUtils.h"
#include "GameFramework/PlayerController.h"
#include "IstanaGameMode.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "UnrealClient.h"
#include "RHI.h"

namespace
{
    bool IntegerField(const FJsonObject& Object, const TCHAR* Name, double Min, double Max)
    {
        double Value;
        return Object.TryGetNumberField(Name, Value) && FMath::IsFinite(Value)
            && Value >= Min && Value <= Max && FMath::FloorToDouble(Value) == Value;
    }
    bool PlacementWireShape(const FJsonObject& Object)
    {
        if (!IntegerField(Object, TEXT("schemaVersion"), 1, 1)
            || !IntegerField(Object, TEXT("revision"), 0, 9007199254740991.0)
            || !IntegerField(Object, TEXT("requestId"), 0, 9007199254740991.0)) return false;
        const TArray<TSharedPtr<FJsonValue>>* Centers = nullptr;
        if (!Object.TryGetArrayField(TEXT("centers"), Centers) || Centers->Num() > 256) return false;
        for (const auto& Value : *Centers)
        {
            const TSharedPtr<FJsonObject>* Center = nullptr;
            const TSharedPtr<FJsonObject>* Position = nullptr;
            if (!Value->TryGetObject(Center) || !IntegerField(**Center, TEXT("groupId"), 0, 255)
                || !(*Center)->TryGetObjectField(TEXT("centerWorldCm"), Position)) return false;
            for (const TCHAR* Axis : {TEXT("x"), TEXT("y"), TEXT("z")})
            { double Coordinate; if (!(*Position)->TryGetNumberField(Axis, Coordinate) || !FMath::IsFinite(Coordinate)) return false; }
        }
        return true;
    }
    FString Encode(const TSharedRef<FJsonObject>& Object)
    {
        FString Text;
        FJsonSerializer::Serialize(Object, TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Text));
        return Text;
    }
}
ARedTeamAgentBridge::ARedTeamAgentBridge() { PrimaryActorTick.bCanEverTick = true; }
void ARedTeamAgentBridge::BeginPlay() { Super::BeginPlay(); if (bStartOnBeginPlay) { FString Error; StartBridge(Error); } }
void ARedTeamAgentBridge::EndPlay(const EEndPlayReason::Type Reason) { StopBridge(); Super::EndPlay(Reason); }
void ARedTeamAgentBridge::Destroyed() { StopBridge(); Super::Destroyed(); }

bool ARedTeamAgentBridge::StartBridge(FString& Error)
{
    Error.Reset();
    if (Listener) return true;
    if (!IsValid(Manager) || Manager->PlacementSource != ERedTeamPlacementSource::AgentPlacement
        || Port < 1024 || Port > 65535 || !FMath::IsFinite(IdleTimeoutSeconds) || IdleTimeoutSeconds < 1)
    { Error = TEXT("Assign an AgentPlacement manager, valid port and positive timeout."); Status = Error; return false; }
    auto* System = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    auto Address = System->CreateInternetAddr(); bool bValid = false;
    Address->SetIp(TEXT("127.0.0.1"), bValid); Address->SetPort(Port);
    Listener = System->CreateSocket(NAME_Stream, TEXT("RedTeamAgentLoopback"), false);
    if (!Listener || !Listener->SetNonBlocking(true) || !Listener->Bind(*Address) || !Listener->Listen(1))
    { StopBridge(); Error = TEXT("Cannot bind red-team loopback port."); Status = Error; return false; }
    LastRequest = -1; LastPayload.Reset(); LastResponse.Reset();
    const FString Directory = FPaths::ProjectSavedDir() / TEXT("RedTeamReplay");
    IFileManager::Get().MakeDirectory(*Directory, true);
    ReplayPath = Directory / (FGuid::NewGuid().ToString() + TEXT(".jsonl"));
    Status = FString::Printf(TEXT("Listening on 127.0.0.1:%d"), Port);
    return true;
}
void ARedTeamAgentBridge::Disconnect(const FString& Reason)
{
    if (Client)
    {
        Client->Close(); ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(Client); Client = nullptr;
        if (IsValid(Manager)) Manager->CancelEpisode(Reason);
    }
    Input.Reset(); Output.Reset(); OutputOffset = 0; Status = Reason;
}
void ARedTeamAgentBridge::StopBridge()
{
    Disconnect(TEXT("Bridge stopped/disconnected"));
    if (Listener) { Listener->Close(); ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->DestroySocket(Listener); Listener = nullptr; }
}

FString ARedTeamAgentBridge::HandleRequest(const FString& Line)
{
    check(IsInGameThread());
    auto Response = MakeShared<FJsonObject>();
    Response->SetBoolField(TEXT("ok"), false);
    TSharedPtr<FJsonObject> Request;
    if (Line.Len() > 65536 || !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Line), Request) || !Request.IsValid())
    { Response->SetStringField(TEXT("error"), TEXT("Invalid JSON or request exceeds 64 KiB.")); return Encode(Response); }
    double Number = -1; FString Op;
    if (!Request->TryGetNumberField(TEXT("id"), Number) || !FMath::IsFinite(Number) || Number < 0
        || Number > 9007199254740991.0 || FMath::FloorToDouble(Number) != Number || !Request->TryGetStringField(TEXT("op"), Op))
    { Response->SetStringField(TEXT("error"), TEXT("Supply integer id and op.")); return Encode(Response); }
    const int64 Id = int64(Number); Response->SetNumberField(TEXT("id"), Number);
    if (Id == LastRequest && Line == LastPayload) return LastResponse;
    if (Id <= LastRequest)
    { Response->SetStringField(TEXT("error"), TEXT("Stale id or conflicting retry.")); return Encode(Response); }
    FString Error;
    if (!IsValid(Manager) || Manager->PlacementSource != ERedTeamPlacementSource::AgentPlacement)
        Error = TEXT("AgentPlacement manager unavailable.");
    else if (Op == TEXT("reset"))
    {
        double Seed;
        FRedTeamPlacementContext Context;
        if (!Request->TryGetNumberField(TEXT("seed"), Seed) || !FMath::IsFinite(Seed) || Seed < MIN_int32 || Seed > MAX_int32 || FMath::FloorToDouble(Seed) != Seed)
            Error = TEXT("seed: expected int32.");
        else if (Manager->BeginPlacementEpisode(int32(Seed), Context, Error))
            Response->SetObjectField(TEXT("context"), FJsonObjectConverter::UStructToJsonObject(Context));
    }
    else if (Op == TEXT("blue_context") || Op == TEXT("blue_observe") || Op == TEXT("blue_deploy"))
    {
        auto* Blue = Manager->BlueCoordinator.Get();
        if (!IsValid(Blue)) Error = TEXT("Blue coordinator unavailable; start Istana with -IstanaBlueLive.");
        else if (Op == TEXT("blue_context")) Response->SetObjectField(TEXT("context"), Blue->ContextJson());
        else if (Op == TEXT("blue_deploy"))
        {
            const TSharedPtr<FJsonObject>* Action = nullptr;
            if (!Request->TryGetObjectField(TEXT("action"), Action)) Error = TEXT("Blue deployment requires an action object.");
            else Response->SetObjectField(TEXT("result"), Blue->DeployJson(**Action, Error));
        }
    }
    else if (Op == TEXT("blue_capture"))
    {
        auto* Blue = Manager->BlueCoordinator.Get();
        auto* PC = GetWorld() ? GetWorld()->GetFirstPlayerController() : nullptr;
        auto* Camera = PC ? Cast<AIstanaCameraPawn>(PC->GetPawn()) : nullptr;
        FString View, RunText; FGuid Run;
        if (!FParse::Param(FCommandLine::Get(), TEXT("IstanaAllowCapture")))
            Error = TEXT("Native capture is opt-in: start with -IstanaAllowCapture.");
        else if (GUsingNullRHI || !IsValid(Blue) || !Camera || !GEngine || !GEngine->GameViewport || !GEngine->GameViewport->Viewport)
            Error = TEXT("Native rendered viewport/camera unavailable (do not use -nullrhi).");
        else if (!Request->TryGetStringField(TEXT("runId"), RunText) || !FGuid::Parse(RunText, Run)
            || Run != Manager->GetPlacementContext().RunId
            || !IntegerField(*Request, TEXT("expectedStep"), 0, 9007199254740991.)
            || Request->GetNumberField(TEXT("expectedStep")) != Manager->GetEpisodeObservation().CompletedSteps)
            Error = TEXT("Capture requires current runId/expectedStep.");
        else if (FScreenshotRequest::IsScreenshotRequested()) Error = TEXT("A screenshot is already pending.");
        else if (!Request->TryGetStringField(TEXT("view"), View) || (View != TEXT("overview") && View != TEXT("layout") && View != TEXT("sensor") && View != TEXT("drone") && View != TEXT("drone_model") && View != TEXT("showcase")))
            Error = TEXT("Capture view must be overview, sensor, drone or drone_model.");
        else
        {
            const FVector Origin = Manager->GetPlacementContext().ObjectiveWorldCm;
            FVector Position = Origin + FVector(10000,-14000,10000), Target = Origin + FVector(0,0,-900);
            if (View == TEXT("layout"))
            {
                Position = Origin + FVector(1000,-4000,15500);
                Target = Origin;
            }
            if (View == TEXT("showcase"))
            {
                double T = 0; FString Angle;
                if (!Request->TryGetNumberField(TEXT("presentationSeconds"), T) || !FMath::IsFinite(T) || T < 0 || T > 120 ||
                    !Request->TryGetStringField(TEXT("showcaseAngle"), Angle) ||
                    (Angle != TEXT("wide") && Angle != TEXT("overhead") && Angle != TEXT("tracking") && Angle != TEXT("sensor") && Angle != TEXT("together")))
                    Error = TEXT("Showcase requires presentationSeconds 0..120 and a valid camera angle.");
                ABlueSensorMarker* Sensor = nullptr;
                for (TActorIterator<ABlueSensorMarker> It(GetWorld()); It; ++It) { Sensor = *It; break; }
                if (!Sensor) Error = TEXT("Showcase needs a display sensor prop.");
                if (Error.IsEmpty())
                {
                    const FVector Base = Sensor->GetActorLocation();
                    const FVector Center = Base + FVector(0,0,480);
                    FVector Lead = Center;
                    double LeadYaw = 0;
                    auto Locations = MakeShared<FJsonObject>();
                    // Animated display props only: not registered as simulated
                    // drones, sensor targets, routes, policies or observations.
                    for (int32 I=0; I<3; ++I)
                    {
                        const FName Tag(*FString::Printf(TEXT("CosmeticFlyingProp%d"), I));
                        AIstanaDroneVisual* Prop = nullptr;
                        for (TActorIterator<AIstanaDroneVisual> It(GetWorld()); It; ++It)
                            if (It->ActorHasTag(Tag)) { Prop = *It; break; }
                        if (!Prop)
                        {
                            FActorSpawnParameters P; P.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
                            Prop = GetWorld()->SpawnActor<AIstanaDroneVisual>(Center, FRotator::ZeroRotator, P);
                            if (Prop) Prop->Tags.Add(Tag);
                        }
                        if (!Prop) { Error = TEXT("Unable to create display prop."); break; }
                        const double A = T*.45 + I*2*PI/3;
                        const FVector Point = Center + FVector(420*FMath::Cos(A),420*FMath::Sin(A),70*FMath::Sin(T*.7+I));
                        const double Yaw = FMath::RadiansToDegrees(A)+90;
                        Prop->SetActorLocationAndRotation(Point, FRotator(-4,Yaw,8*FMath::Sin(A)));
                        Prop->AnimateDisplayRotors(T + I*.13);
                        Locations->SetStringField(FString::FromInt(I), Point.ToString());
                        if (I == 0) { Lead = Point; LeadYaw = Yaw; }
                    }
                    // Visible scanning is cosmetic rotation, not a sensing run.
                    Sensor->SetActorRotation(FRotator(0,55*FMath::Sin(T*.8),0));
                    Target = Center;
                    if (Angle == TEXT("wide")) Position = Base + FVector(2300,-2800,2000);
                    else if (Angle == TEXT("overhead")) Position = Center + FVector(0,-30,1800);
                    else if (Angle == TEXT("tracking"))
                    { Target = Lead; Position = Lead + FRotator(0,LeadYaw,0).RotateVector(FVector(-260,-200,110)); }
                    else if (Angle == TEXT("sensor"))
                    { Target = Base + FVector(0,0,230); Position = Base + FVector(600,-700,350); }
                    else Position = Base + FVector(850,-1050,900);
                    Response->SetObjectField(TEXT("cosmeticPropLocationsCm"), Locations);
                    Response->SetNumberField(TEXT("presentationSeconds"), T);
                    Response->SetBoolField(TEXT("cosmeticMotionOnly"), true);
                }
            }
            if (View == TEXT("drone_model"))
            {
                // Static cosmetic display prop, never added to the solver,
                // agent observations, sensor targets or movement manager.
                AIstanaDroneVisual* Prop = nullptr;
                for (TActorIterator<AIstanaDroneVisual> It(GetWorld()); It; ++It)
                    if (It->ActorHasTag(TEXT("CaptureOnlyDroneModel"))) { Prop = *It; break; }
                Target = Origin + FVector(0,-1000,400);
                if (!Prop)
                {
                    FActorSpawnParameters Parameters;
                    Parameters.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
                    Prop = GetWorld()->SpawnActor<AIstanaDroneVisual>(Target, FRotator::ZeroRotator, Parameters);
                    if (Prop) Prop->Tags.Add(TEXT("CaptureOnlyDroneModel"));
                }
                if (!Prop) Error = TEXT("Unable to create cosmetic display model.");
                Position = Target + FVector(200,-220,120);
            }
            if (View == TEXT("drone"))
            {
                const auto States = Manager->GetDroneStates();
                const auto* Drone = States.FindByPredicate([](const auto& State) { return State.DroneId == 0; });
                if (!Drone) Error = TEXT("Drone close-up requires a spawned episode.");
                else
                {
                    const FVector Forward = (Origin-Drone->PositionCm).GetSafeNormal2D();
                    const FVector Side(-Forward.Y, Forward.X, 0);
                    Target = Drone->PositionCm;
                    Position = Target - Forward*200 + Side*220 + FVector(0,0,120);
                }
            }
            if (View == TEXT("sensor"))
            {
                if (!IntegerField(*Request, TEXT("siteId"), 0, Blue->ApprovedSitesM.Num()-1)) Error = TEXT("Invalid capture siteId.");
                else
                {
                    const auto Site = Blue->ApprovedSitesM[int32(Request->GetNumberField(TEXT("siteId")))];
                    ABlueSensorMarker* Selected = nullptr;
                    for (TActorIterator<ABlueSensorMarker> It(GetWorld()); It; ++It)
                        if (FVector2D::Distance(FVector2D(It->GetActorLocation()-Origin), Site*100) < 1.) { Selected = *It; break; }
                    if (!Selected) Error = TEXT("Capture site has no deployed sensor.");
                    else
                    {
                        Target = Selected->GetActorLocation() + FVector(0,0,215);
                        Position = Selected->GetActorLocation() + Selected->GetActorRotation().RotateVector(FVector(640,-760,360));
                    }
                }
            }
            if (Error.IsEmpty())
            {
                // Presentation-only orbit around the existing camera target.
                // Does not move any simulated object or advance the solver.
                double OrbitDegrees = 0;
                if (Request->HasField(TEXT("orbitDegrees")) &&
                    (!Request->TryGetNumberField(TEXT("orbitDegrees"), OrbitDegrees) ||
                     !FMath::IsFinite(OrbitDegrees) || FMath::Abs(OrbitDegrees) > 45.))
                    Error = TEXT("Capture orbitDegrees must be finite and within +/-45.");
                if (!Error.IsEmpty())
                {
                    Response->SetBoolField(TEXT("ok"), false);
                    Response->SetStringField(TEXT("error"), Error);
                    return Encode(Response);
                }
                Position = Target + (Position-Target).RotateAngleAxis(OrbitDegrees, FVector::UpVector);
                Blue->bDrawCoverage = false;
                // Raw footage by default: debug points otherwise hide the actual
                // drone and sensor meshes. Labels belong outside the video.
                Blue->bCapturePresentation = false;
                Manager->bDrawDebug = false;
                Camera->SetView(3); // disables touring/velocity before applying a fixed capture camera
                Camera->SetActorLocationAndRotation(Position, (Target-Position).Rotation());
                Response->SetStringField(TEXT("cameraLocationCm"), Camera->GetViewLocation().ToString());
                Response->SetStringField(TEXT("cameraTargetCm"), Target.ToString());
                Response->SetStringField(TEXT("cameraRotation"), Camera->GetViewRotation().ToString());
                // Presentation anchors only; no coverage or performance values.
                // Callers warm the camera before retaining a capture.
                if (View == TEXT("layout"))
                {
                    TArray<TSharedPtr<FJsonValue>> Anchors;
                    if (PC)
                    {
                        for (TActorIterator<ABlueSensorMarker> It(GetWorld()); It; ++It)
                        {
                            FVector2D Pixel;
                            if (PC->ProjectWorldLocationToScreen(It->GetActorLocation(), Pixel))
                            {
                                auto Anchor = MakeShared<FJsonObject>();
                                Anchor->SetNumberField(TEXT("x"), Pixel.X);
                                Anchor->SetNumberField(TEXT("y"), Pixel.Y);
                                Anchors.Add(MakeShared<FJsonValueObject>(Anchor));
                            }
                        }
                    }
                    Response->SetArrayField(TEXT("sensorScreenAnchors"), Anchors);
                }
                if (View == TEXT("drone"))
                {
                    double Closest = TNumericLimits<double>::Max();
                    for (TActorIterator<AIstanaDroneVisual> It(GetWorld()); It; ++It)
                        if (!It->IsHidden()) Closest = FMath::Min(Closest, FVector::Dist(It->GetActorLocation(), Target));
                    Response->SetNumberField(TEXT("nearestVisibleDroneActorToTargetCm"), Closest);
                }
                const FString Relative = FString::Printf(TEXT("BlueCapture/%s/%lld-%lld.png"), *Run.ToString(), Manager->GetEpisodeObservation().CompletedSteps, Id);
                const FString Destination = FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir()/Relative);
                IFileManager::Get().MakeDirectory(*FPaths::GetPath(Destination), true);
                if (IFileManager::Get().FileExists(*Destination)) Error = TEXT("Capture destination already exists; never overwrite evidence.");
                else
                {
                    FScreenshotRequest::RequestScreenshot(Destination, false, false, false);
                    Response->SetStringField(TEXT("captureRelativeToSaved"), Relative);
                    Response->SetStringField(TEXT("captureStatus"), TEXT("requested; wait for PNG completion before advancing"));
                }
            }
        }
    }
    else if (Op == TEXT("context"))
        Response->SetObjectField(TEXT("context"), FJsonObjectConverter::UStructToJsonObject(Manager->GetPlacementContext()));
    else if (Op == TEXT("place"))
    {
        const TSharedPtr<FJsonObject>* Object = nullptr; FRedTeamPlacementAction Action; FText Reason;
        if (!Request->TryGetObjectField(TEXT("action"), Object) || !PlacementWireShape(**Object)
            || !FJsonObjectConverter::JsonObjectToUStruct((*Object).ToSharedRef(), &Action, 0, 0, true, &Reason))
            Error = TEXT("action: invalid placement schema. ") + Reason.ToString();
        else
        {
            auto Result = Manager->SubmitPlacement(Action);
            Response->SetObjectField(TEXT("result"), FJsonObjectConverter::UStructToJsonObject(Result));
            Error = Result.Error;
        }
    }
    else if (Op == TEXT("step"))
    {
        double Steps; FString RunText; FGuid Run; double Expected;
        if (!Request->TryGetStringField(TEXT("runId"), RunText) || !FGuid::Parse(RunText, Run)
            || Run != Manager->GetPlacementContext().RunId || !Request->TryGetNumberField(TEXT("expectedStep"), Expected)
            || Expected != Manager->GetEpisodeObservation().CompletedSteps)
            Error = TEXT("Stale runId/expectedStep.");
        else if (!Request->TryGetNumberField(TEXT("steps"), Steps) || !FMath::IsFinite(Steps) || Steps < 1 || Steps > 1000 || FMath::FloorToDouble(Steps) != Steps)
            Error = TEXT("steps: expected integer 1..1000.");
        else
        {
            FRedTeamEpisodeObservation Observation;
            Manager->AdvanceEpisode(int32(Steps), Observation, Error);
            Response->SetObjectField(TEXT("observation"), FJsonObjectConverter::UStructToJsonObject(Observation));
        }
    }
    else if (Op == TEXT("observe"))
        Response->SetObjectField(TEXT("observation"), FJsonObjectConverter::UStructToJsonObject(Manager->GetEpisodeObservation()));
    else if (Op == TEXT("cancel")) Manager->CancelEpisode(TEXT("External agent cancelled episode"));
    else Error = TEXT("Unknown op; use reset/context/place/step/observe/cancel/blue_context/blue_deploy/blue_observe.");
    if (IsValid(Manager) && IsValid(Manager->BlueCoordinator))
        Response->SetObjectField(TEXT("blueObservation"), Manager->BlueCoordinator->ObservationJson());
    Response->SetBoolField(TEXT("ok"), Error.IsEmpty());
    if (!Error.IsEmpty()) Response->SetStringField(TEXT("error"), Error);
    LastRequest = Id; LastPayload = Line; LastResponse = Encode(Response);
    if (!ReplayPath.IsEmpty())
    {
        auto Record = MakeShared<FJsonObject>(); Record->SetObjectField(TEXT("request"), Request);
        Record->SetObjectField(TEXT("response"), Response);
        if (!FFileHelper::SaveStringToFile(Encode(Record) + TEXT("\n"), *ReplayPath,
            FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM, &IFileManager::Get(), FILEWRITE_Append))
            Status = TEXT("Replay log write failed");
    }
    return LastResponse;
}

void ARedTeamAgentBridge::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);
    if (!Listener) return;
    if (!Client)
    {
        bool bPending = false;
        if (Listener->HasPendingConnection(bPending) && bPending)
        {
            Client = Listener->Accept(TEXT("RedTeamAgent"));
            if (Client)
            {
                Client->SetNonBlocking(true); LastActivity = FPlatformTime::Seconds(); Status = TEXT("Agent connected");
                LastRequest = -1; LastPayload.Reset(); LastResponse.Reset();
            }
        }
        return;
    }
    if (FPlatformTime::Seconds() - LastActivity > IdleTimeoutSeconds)
    { Disconnect(TEXT("Agent idle timeout; episode cancelled")); return; }
    // At most 64 KiB received and one operation dispatched per tick; no blocking network calls.
    uint32 Pending = 0;
    if (Output.IsEmpty() && Client->HasPendingData(Pending) && Pending)
    {
        uint8 Buffer[8192]; int32 Read = 0;
        if (Client->Recv(Buffer, int32(FMath::Min(Pending, uint32(8192))), Read) && Read > 0)
        { Input.Append(Buffer, Read); LastActivity = FPlatformTime::Seconds(); }
        if (Input.Num() > 65536) { Disconnect(TEXT("Request exceeds 64 KiB")); return; }
    }
    if (Output.IsEmpty())
    {
        int32 Newline = Input.IndexOfByKey(uint8('\n'));
        if (Newline != INDEX_NONE)
        {
            FUTF8ToTCHAR Converted(reinterpret_cast<const ANSICHAR*>(Input.GetData()), Newline);
            const FString Response = HandleRequest(FString(Converted.Length(), Converted.Get())) + TEXT("\n");
            Input.RemoveAt(0, Newline + 1);
            FTCHARToUTF8 Encoded(*Response); Output.Append(reinterpret_cast<const uint8*>(Encoded.Get()), Encoded.Length());
            LastActivity = FPlatformTime::Seconds();
        }
    }
    if (!Output.IsEmpty())
    {
        int32 Sent = 0;
        if (Client->Send(Output.GetData() + OutputOffset, Output.Num() - OutputOffset, Sent)) OutputOffset += Sent;
        if (OutputOffset == Output.Num()) { Output.Reset(); OutputOffset = 0; }
    }
    uint8 PeekByte; int32 PeekRead = -1;
    const bool bPeek = Client->Recv(&PeekByte, 1, PeekRead, ESocketReceiveFlags::Peek);
    if ((!bPeek && PeekRead == 0) || Client->GetConnectionState() == SCS_ConnectionError) Disconnect(TEXT("Agent disconnected; episode cancelled"));
}
