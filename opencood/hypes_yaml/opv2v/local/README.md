# OPV2V Local Modalities

This directory contains the OPV2V local / single-modality configuration files. The directory layout is:

```text
local/
  <model family>/
    <method folder>/
      config.yaml
```

Each second-level subfolder represents one concrete model-modality method. The `ego_modality` field in `config.yaml` is the modality ID used by this codebase. In the paper, these modalities are renumbered consecutively from `m1` to `m30` for presentation, so the paper IDs and repository IDs are no longer identical after repository ID `m20`.

In the table below, "Repository Modality ID" corresponds to the left side of the figure, and "Paper Modality ID" corresponds to the consecutive `m1` to `m30` notation used in the paper.

| No. | Method Folder | Repository Modality ID | Paper Modality ID | Sensor | Encoder / Backbone | Voxel size (m) | Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `LSSeff/LSSeff48` | `m1` | `m1` | Camera | LSS (EfficientNet-B0) | - | - |
| 2 | `SECOND/local_sd1` | `m2` | `m2` | LiDAR | SECOND | `[0.1, 0.1, 0.1]` | Normal |
| 3 | `LSSres/LSSres101` | `m3` | `m3` | Camera | LSS (ResNet-101) | - | - |
| 4 | `PointPillar/local_pp4` | `m4` | `m4` | LiDAR | PointPillar | `[0.4, 0.4, 0.4]` | Normal |
| 5 | `VoxelNet/local_vn4` | `m5` | `m5` | LiDAR | VoxelNet | `[0.4, 0.4, 0.4]` | Normal |
| 6 | `PointPillar/local_pp4_medium` | `m6` | `m6` | LiDAR | PointPillar | `[0.4, 0.4, 0.4]` | Medium |
| 7 | `PointPillar/local_pp4_large` | `m7` | `m7` | LiDAR | PointPillar | `[0.4, 0.4, 0.4]` | Large |
| 8 | `SECOND/local_sd1_medium` | `m8` | `m8` | LiDAR | SECOND | `[0.1, 0.1, 0.1]` | Medium |
| 9 | `SECOND/local_sd1_large` | `m9` | `m9` | LiDAR | SECOND | `[0.1, 0.1, 0.1]` | Large |
| 10 | `VoxelNet/local_vn4_medium` | `m10` | `m10` | LiDAR | VoxelNet | `[0.4, 0.4, 0.4]` | Medium |
| 11 | `VoxelNet/local_vn4_large` | `m11` | `m11` | LiDAR | VoxelNet | `[0.4, 0.4, 0.4]` | Large |
| 12 | `PointPillar/local_pp8` | `m12` | `m12` | LiDAR | PointPillar | `[0.8, 0.8, 0.8]` | Normal |
| 13 | `PointPillar/local_pp8_medium` | `m13` | `m13` | LiDAR | PointPillar | `[0.8, 0.8, 0.8]` | Medium |
| 14 | `PointPillar/local_pp8_large` | `m14` | `m14` | LiDAR | PointPillar | `[0.8, 0.8, 0.8]` | Large |
| 15 | `SECOND/local_sd2` | `m15` | `m15` | LiDAR | SECOND | `[0.2, 0.2, 0.2]` | Normal |
| 16 | `SECOND/local_sd2_medium` | `m16` | `m16` | LiDAR | SECOND | `[0.2, 0.2, 0.2]` | Medium |
| 17 | `SECOND/local_sd2_large` | `m17` | `m17` | LiDAR | SECOND | `[0.2, 0.2, 0.2]` | Large |
| 18 | `PointPillar/local_pp2` | `m18` | `m18` | LiDAR | PointPillar | `[0.2, 0.2, 0.2]` | Normal |
| 19 | `PointPillar/local_pp2_medium` | `m19` | `m19` | LiDAR | PointPillar | `[0.2, 0.2, 0.2]` | Medium |
| 20 | `PointPillar/local_pp2_large` | `m20` | `m20` | LiDAR | PointPillar | `[0.2, 0.2, 0.2]` | Large |
| 21 | `SECOND/local_sd4` | `m24` | `m21` | LiDAR | SECOND | `[0.4, 0.4, 0.4]` | Normal |
| 22 | `SECOND/local_sd4_medium` | `m25` | `m22` | LiDAR | SECOND | `[0.4, 0.4, 0.4]` | Medium |
| 23 | `SECOND/local_sd4_large` | `m26` | `m23` | LiDAR | SECOND | `[0.4, 0.4, 0.4]` | Large |
| 24 | `VoxelNet/local_vn8` | `m30` | `m24` | LiDAR | VoxelNet | `[0.8, 0.8, 0.8]` | Normal |
| 25 | `VoxelNet/local_vn8_medium` | `m31` | `m25` | LiDAR | VoxelNet | `[0.8, 0.8, 0.8]` | Medium |
| 26 | `VoxelNet/local_vn8_large` | `m32` | `m26` | LiDAR | VoxelNet | `[0.8, 0.8, 0.8]` | Large |
| 27 | `LSSeff/LSSeffB1` | `m36` | `m27` | Camera | LSS (EfficientNet-B1) | - | - |
| 28 | `LSSres/LSSres34` | `m37` | `m28` | Camera | LSS (ResNet-34) | - | - |
| 29 | `LSSeff/LSSeffB2` | `m38` | `m29` | Camera | LSS (EfficientNet-B2) | - | - |
| 30 | `LSSres/LSSres50` | `m39` | `m30` | Camera | LSS (ResNet-50) | - | - |


