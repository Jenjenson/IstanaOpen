#pragma once
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RedTeamAgentBridge.generated.h"
class ARedTeamManager;
class FSocket;

/** Opt-in loopback JSON-lines adapter. Inference runs in the external process; all UE work stays on the game thread. */
UCLASS(BlueprintType, Blueprintable)
class ISTANAOPEN_API ARedTeamAgentBridge : public AActor
{
    GENERATED_BODY()
public:
    ARedTeamAgentBridge();
    UPROPERTY(EditInstanceOnly, BlueprintReadWrite, Category="Red Team|Bridge") TObjectPtr<ARedTeamManager> Manager;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Red Team|Bridge") bool bStartOnBeginPlay = false;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Red Team|Bridge", meta=(ClampMin="1024", ClampMax="65535")) int32 Port = 8765;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category="Red Team|Bridge", meta=(ClampMin="1")) double IdleTimeoutSeconds = 60;
    UPROPERTY(BlueprintReadOnly, Category="Red Team|Bridge") FString Status;
    UFUNCTION(BlueprintCallable, Category="Red Team|Bridge") bool StartBridge(FString& Error);
    UFUNCTION(BlueprintCallable, Category="Red Team|Bridge") void StopBridge();
    // Transport-neutral request dispatcher, also used by in-process integrations/tests.
    FString HandleRequest(const FString& Line);
    virtual void Tick(float DeltaSeconds) override;
protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type Reason) override;
    virtual void Destroyed() override;
private:
    void Disconnect(const FString& Reason);
    FSocket* Listener = nullptr;
    FSocket* Client = nullptr;
    TArray<uint8> Input, Output;
    int32 OutputOffset = 0;
    double LastActivity = 0;
    int64 LastRequest = -1;
    FString LastPayload, LastResponse, ReplayPath;
};
