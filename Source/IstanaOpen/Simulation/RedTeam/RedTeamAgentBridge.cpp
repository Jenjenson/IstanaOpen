#include "Simulation/RedTeam/RedTeamAgentBridge.h"
#include "Simulation/RedTeam/RedTeamManager.h"
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
    else Error = TEXT("Unknown op; use reset/context/place/step/observe/cancel.");
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
