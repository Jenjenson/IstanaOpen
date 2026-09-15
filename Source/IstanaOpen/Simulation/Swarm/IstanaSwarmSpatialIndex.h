#pragma once
#include "CoreMinimal.h"
#include "Simulation/IstanaSimulationTypes.h"

/** Broad phase only. Callers retain exact distance predicates and accumulation order. */
class FIstanaSwarmSpatialIndex
{
    struct FCell
    {
        int64 X, Y, Z;
        bool operator==(const FCell& B) const { return X == B.X && Y == B.Y && Z == B.Z; }
        friend uint32 GetTypeHash(const FCell& C)
        { return HashCombine(HashCombine(::GetTypeHash(C.X), ::GetTypeHash(C.Y)), ::GetTypeHash(C.Z)); }
    };
    TMap<FCell, TArray<int32>> Cells;
    double Width = 1;
    int32 Count = 0;
    bool bFallback = false;
    FCell Key(const FVector& P) const
    { return {int64(FMath::FloorToDouble(P.X / Width)), int64(FMath::FloorToDouble(P.Y / Width)), int64(FMath::FloorToDouble(P.Z / Width))}; }
public:
    void Build(const TArray<FIstanaDroneState>& States, double Radius)
    {
        Cells.Reset(); Width = 2 * Radius; Count = States.Num(); bFallback = false;
        for (const auto& S : States)
            if (S.PositionCm.ContainsNaN() || S.PositionCm.GetAbsMax() / Width > 1.e12) bFallback = true;
        if (bFallback) return; // Preserve behavior even at coordinates outside safe integer conversion.
        for (int32 I = 0; I < Count; ++I)
            if (States[I].bActive) Cells.FindOrAdd(Key(States[I].PositionCm)).Add(I);
        if (Cells.Num() <= 8) bFallback = true; // Dense populations favor the ordered direct scan.
    }
    void Candidates(const FVector& P, TArray<int32>& Out) const
    {
        Out.Reset();
        if (bFallback) { for (int32 I = 0; I < Count; ++I) Out.Add(I); return; }
        const FCell C = Key(P);
        // Double-radius cells leave room for floating-point boundary rounding.
        for (int64 X = -1; X <= 1; ++X)
            for (int64 Y = -1; Y <= 1; ++Y)
                for (int64 Z = -1; Z <= 1; ++Z)
                    if (const auto* Bucket = Cells.Find({C.X + X, C.Y + Y, C.Z + Z})) Out.Append(*Bucket);
        Out.Sort();
    }
};
