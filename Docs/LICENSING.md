# Distribution and asset licensing

Reviewed against official sources on 10 September 2026.

Istana Open uses a public-source project and a separately packaged Unreal application. Its own source and assets can be published on GitHub under their stated licences. Unreal Engine itself remains proprietary: this project must not be described as having no licensing conditions whatsoever.

## What belongs in the public repository

- Original project code, build scripts, procedural geometry, materials, and project settings, under the root project licence.
- Individually identified Poly Haven and ambientCG asset files and their derivatives, under CC0-1.0. Include source URLs and checksums in the asset manifest.
- The OpenStreetMap source extract and derived geographic data, kept identifiable as ODbL-1.0 data with contributor attribution. These are not covered by the root code licence.
- Project-owned Unreal assets created from these admitted sources.

Keep the Unreal installation, Engine source, Editor binaries, intermediate build files, caches, authentication tokens, and unrelated assets out of the public repository. Do not copy the preceding TRIAD project's Content directory wholesale. Its weather, Starter Content, and third-party scene packs have separate rights and are not admitted by this project's asset policy.

## Unreal source project and playable release

Developers obtain Unreal separately from Epic and accept its EULA. The public project contains references to Unreal APIs and engine resources, not a redistribution of Epic's editor or engine source. Epic permits distribution of a finished Product with engine code incorporated as inseparable object code; engine source and Engine Tools have separate distribution restrictions. See sections 4 and 5 of the [Unreal Engine EULA](https://www.unrealengine.com/eula/unreal).

Judges should download the packaged Windows release, extract it, and run its launcher; they do not need the Unreal Editor. The package must retain the engine and third-party notices emitted by Unreal's packaging process and the supplied end-user terms. Other operating systems require their own tested builds. Availability on GitHub does not turn the bundled Unreal runtime into MIT-licensed code.

The [Epic Content License Agreement](https://www.unrealengine.com/eula/content), particularly sections 3 and 4, distinguishes use of licensed content inside a finished project from distribution of source-format assets. A marketplace purchase, free download, or a working local project is not by itself permission to put that content in a public source repository.

## Natural assets

[Poly Haven's asset licence](https://polyhaven.com/license) permits redistribution and commercial use of its downloadable models, textures, and HDRIs under [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/). Its website text, logos, and example renders have different terms. Copy asset files and factual provenance, not archived website pages or marketing imagery.

[ambientCG's licence](https://docs.ambientcg.com/license/) also places downloadable assets under CC0-1.0 and permits inclusion of raw files. Source credit is retained for traceability even when CC0 does not require attribution. Generic botanical assets represent appearance; they do not establish exact species or individual-tree locations at the Istana.

## OpenStreetMap

Credit the data visibly in the application and documentation as **© OpenStreetMap contributors**, with links to [OpenStreetMap copyright and licence](https://www.openstreetmap.org/copyright) and the [ODbL 1.0 licence](https://opendatacommons.org/licenses/odbl/1-0/). Retain the source extract, acquisition date, query or bounds, and transformation scripts with the release source. Preserve ODbL terms for distributed extracts and adapted databases. This does not automatically place independent application code under ODbL.

Footprints and mapped tags provide approximate geographic context. Unspecified building heights, facade details, and terrain treatments are authored approximations, not measured site data.

## Why the offline build does not depend on streamed Cesium data

Microsoft's original [AirSim licence](https://github.com/microsoft/AirSim/blob/main/LICENSE) is MIT and permits redistribution with its notice. AirSim itself was therefore not a blanket prohibition on submission. Removing it simplifies this standalone architectural experience and does not resolve the rights of unrelated content automatically.

The [Cesium for Unreal FAQ](https://cesium.com/learn/unreal/unreal-faq/) states that its plugin is Apache-2.0 and can load local tilesets. That licence does not grant ownership or offline redistribution rights over streamed data. Cesium ion's commercial datasets have separate terms; a downloaded plugin is not a high-detail Singapore dataset.

[Google Maps Platform terms](https://cloud.google.com/maps-platform/terms), section 3.2.3, restrict extracting, caching, rehosting, and making content from Google Maps data. Google Photorealistic 3D Tiles, Google satellite images, and Street View content are not bundled or used to generate this offline repository. OneMap basemap pixels and other unlicensed imagery are likewise not admitted.

Original building geometry, CC0 nature assets, and attributed OSM context are the chosen offline approach. If another dataset is added, its specific licence and redistribution rights must be recorded before inclusion; an open rendering API or public viewing website alone is not sufficient.
