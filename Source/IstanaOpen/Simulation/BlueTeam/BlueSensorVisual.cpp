#include "Simulation/BlueTeam/BlueTeamCoordinator.h"
#include "Components/SceneComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInterface.h"
#include "UObject/ConstructorHelpers.h"

ABlueSensorMarker::ABlueSensorMarker()
{
    PrimaryActorTick.bCanEverTick = false;
    SetRootComponent(CreateDefaultSubobject<USceneComponent>(TEXT("SurfaceMount")));
    static ConstructorHelpers::FObjectFinder<UStaticMesh> Cube(TEXT("/Engine/BasicShapes/Cube.Cube"));
    static ConstructorHelpers::FObjectFinder<UStaticMesh> Cylinder(TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
    static ConstructorHelpers::FObjectFinder<UStaticMesh> Sphere(TEXT("/Engine/BasicShapes/Sphere.Sphere"));
    static ConstructorHelpers::FObjectFinder<UMaterialInterface> Metal(TEXT("/Game/Open/Materials/M_metal.M_metal"));
    static ConstructorHelpers::FObjectFinder<UMaterialInterface> Paint(TEXT("/Game/Open/Materials/M_white.M_white"));
    static ConstructorHelpers::FObjectFinder<UMaterialInterface> Lens(TEXT("/Game/Open/Materials/M_glass.M_glass"));
    static ConstructorHelpers::FObjectFinder<UMaterialInterface> Rubber(TEXT("/Game/Open/Materials/M_urbanroof.M_urbanroof"));
    CubeMesh = Cube.Object; CylinderMesh = Cylinder.Object; SphereMesh = Sphere.Object;
    MetalMaterial = Metal.Object; PaintMaterial = Paint.Object; LensMaterial = Lens.Object; RubberMaterial = Rubber.Object;
}

UStaticMeshComponent* ABlueSensorMarker::Part(UStaticMesh* Mesh, UMaterialInterface* Material,
    const FVector& Position, const FVector& SizeCm, const FRotator& Rotation)
{
    auto* Component = NewObject<UStaticMeshComponent>(this);
    Component->SetupAttachment(RootComponent);
    Component->SetMobility(EComponentMobility::Movable);
    Component->SetStaticMesh(Mesh); Component->SetMaterial(0, Material);
    Component->SetRelativeLocationAndRotation(Position, Rotation);
    Component->SetRelativeScale3D(SizeCm / 100.);
    Component->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    Component->SetCanEverAffectNavigation(false); Component->SetCastShadow(true);
    Component->bReceivesDecals = false;
    Component->RegisterComponent(); Parts.Add(Component);
    return Component;
}

void ABlueSensorMarker::Strut(const FVector& A, const FVector& B, double DiameterCm)
{
    const FVector D = B - A;
    Part(CylinderMesh, MetalMaterial, (A+B)*.5, FVector(DiameterCm, DiameterCm, D.Size()), FRotationMatrix::MakeFromZ(D).Rotator());
}

void ABlueSensorMarker::SetMastHeight(double HeightM) { ConfigureSensor(TEXT("eo"), HeightM); }

void ABlueSensorMarker::ConfigureSensor(const FString& ProfileId, double HeightM)
{
    for (const auto& Component : Parts) if (IsValid(Component)) Component->DestroyComponent();
    Parts.Reset();
    const double H = FMath::Max(30., HeightM * 100.);
    // All hardware fits inside the already-validated 80 cm mount footprint.
    // Cosmetic parts have NO collision, navigation, clock, reward or RNG effects.
    Part(CubeMesh, MetalMaterial, FVector(0,0,5), FVector(60,60,10));
    for (double X : {-23.,23.}) for (double Y : {-23.,23.})
    {
        Part(CylinderMesh, RubberMaterial, FVector(X,Y,2), FVector(14,14,4));
        Part(CylinderMesh, MetalMaterial, FVector(X,Y,12), FVector(6,6,6));
    }
    Part(CylinderMesh, MetalMaterial, FVector(0,0,21), FVector(26,26,24));
    Part(CylinderMesh, PaintMaterial, FVector(0,0,H*.25), FVector(16,16,H*.5));
    Part(CylinderMesh, MetalMaterial, FVector(0,0,H*.625), FVector(12,12,H*.25));
    Part(CylinderMesh, PaintMaterial, FVector(0,0,H*.875-12), FVector(9,9,FMath::Max(1.,H*.25-24)));
    for (double Z : {H*.48,H*.74})
        Part(CylinderMesh, RubberMaterial, FVector(0,0,Z), FVector(19,19,7));
    for (int32 I=0; I<3; ++I)
    {
        const double Angle = 2*PI*I/3.;
        Strut(FVector(26*FMath::Cos(Angle),26*FMath::Sin(Angle),12), FVector(0,0,FMath::Min(110.,H*.4)), 4);
    }
    // Sealed power/network enclosure, lid, cable conduit and strain-relief loops.
    Part(CubeMesh, PaintMaterial, FVector(20,0,48), FVector(20,32,48));
    Part(CubeMesh, MetalMaterial, FVector(30.5,0,48), FVector(2,28,42));
    for (double Z : {33.,62.}) Part(CubeMesh, RubberMaterial, FVector(32,0,Z), FVector(3,8,3));
    Strut(FVector(-8,0,42),FVector(-8,0,H-27),2.5);
    for (double Z : {H*.32,H*.60,H*.82})
        Part(CubeMesh, MetalMaterial,FVector(-8,0,Z),FVector(6,5,4));

    Part(CylinderMesh, MetalMaterial, FVector(0,0,H-20), FVector(24,24,20));
    Part(CylinderMesh, RubberMaterial, FVector(0,0,H-10), FVector(27,27,4));
    const FRotator Face(90,0,0); // cylinders' local Z axis faces outward (+X)
    auto Camera = [&](double Y, bool Thermal)
    {
        // Rounded gimbal, weather hood, separate lens barrel, glass and trim.
        Part(SphereMesh, PaintMaterial, FVector(0,Y,H), FVector(28,30,28));
        Part(CubeMesh, PaintMaterial, FVector(11,Y,H+5), FVector(34,27,23));
        Part(CubeMesh, MetalMaterial, FVector(12,Y,H+18), FVector(43,33,3));
        Part(CylinderMesh, RubberMaterial, FVector(29,Y,H+5), FVector(22,22,12), Face);
        Part(CylinderMesh, MetalMaterial, FVector(35,Y,H+5), FVector(24,24,3), Face);
        Part(CylinderMesh, LensMaterial, FVector(37,Y,H+5), FVector(Thermal?17:19,Thermal?17:19,1), Face);
        if (Thermal)
        {
            Part(CylinderMesh, RubberMaterial, FVector(29,Y-9,H-9), FVector(8,8,7), Face);
            Part(CylinderMesh, LensMaterial, FVector(33,Y-9,H-9), FVector(6,6,1), Face);
        }
        for (double Side : {-1.,1.})
        {
            Part(CubeMesh, MetalMaterial, FVector(0,Y+Side*17,H-4), FVector(10,4,30));
            Part(CylinderMesh, RubberMaterial, FVector(0,Y+Side*20,H), FVector(11,11,4), FRotator(0,0,90));
        }
    };
    auto Radar = [&](double Y)
    {
        Part(CubeMesh, MetalMaterial, FVector(-3,Y,H+12), FVector(15,56,50));
        Part(CubeMesh, PaintMaterial, FVector(6,Y,H+12), FVector(8,53,47));
        Part(CubeMesh, RubberMaterial, FVector(10.5,Y,H+12), FVector(1,47,41));
        Part(CubeMesh, PaintMaterial, FVector(11.5,Y,H+12), FVector(1,44,38));
        for (int32 I=0; I<6; ++I)
            Part(CubeMesh, MetalMaterial, FVector(-12,Y,H-6+I*7), FVector(6,45,2));
        for (double Side : {-1.,1.})
            Part(CylinderMesh, MetalMaterial, FVector(12,Y+Side*24,H+32), FVector(3,3,2), Face);
        Strut(FVector(-10,Y,H-10),FVector(-4,Y,H-28),8);
    };
    if (ProfileId == TEXT("rf"))
    {
        Part(CylinderMesh, PaintMaterial, FVector(0,0,H+8), FVector(23,23,48));
        Part(SphereMesh, PaintMaterial, FVector(0,0,H+32), FVector(23,23,12));
        for (int32 I=0; I<4; ++I)
        {
            const double A = PI*.5*I;
            const FVector Base(26*FMath::Cos(A),26*FMath::Sin(A),H-10);
            Strut(FVector(0,0,H-15),Base,3);
            Part(CylinderMesh, RubberMaterial, Base+FVector(0,0,22), FVector(4,4,44));
            Part(SphereMesh, RubberMaterial, Base+FVector(0,0,45), FVector(4,4,4));
        }
    }
    else if (ProfileId == TEXT("radar")) Radar(0);
    else if (ProfileId == TEXT("fused"))
    {
        // Stacked, not widened, to stay inside the approved support footprint.
        Radar(0);
        const double SavedH = H; // lower camera pod attached to the mast
        Part(CubeMesh, MetalMaterial, FVector(16,0,SavedH-70), FVector(36,8,8));
        Part(CubeMesh, PaintMaterial, FVector(18,0,SavedH-60), FVector(32,26,24));
        Part(CylinderMesh, RubberMaterial, FVector(36,0,SavedH-60), FVector(22,22,6), Face);
        Part(CylinderMesh, LensMaterial, FVector(39.5,0,SavedH-60), FVector(17,17,1), Face);
    }
    else Camera(0, ProfileId == TEXT("thermal"));
}
