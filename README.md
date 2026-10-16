# YOLO26 for MyeLiN

This repository contains a PLeX/MyeLiN port of the YOLO26 architecture.

It does not claim authorship or ownership of YOLO26, its model architecture,
weights, name, or related materials.

## Supported workflow

The supplied standalone manifest compiles YOLO26n with F32 storage for CUDA.
In a composed application, the root project may select a different backend and
storage policy for the Yolo graph path.

The input shape is fixed to `1 x 3 x 640 x 640`. Image preprocessing uses
OpenCV and the artifact returns one encoded detection result per invocation.

## Requirements

- MyeLiN and its submodules;
- a C++23 compiler;
- wasi-sdk;
- OpenCV libraries;
- a compatible `yolo26n.safetensors` checkpoint;
- CUDA tools only for CUDA variants.

Model weights are not provided by the project license.

## Add the package

Declare Yolo26 and the weight format selected by the application:

```toml
[dependencies]
yolo26 = { git = "https://github.com/useplx/Yolo26.git", version = "v1.0.0" }
```

Yolo26 exposes three package parts:

| Name | Purpose |
|---|---|
| `yolo26:graph` | Model graph consumed by the SYNaPSe compiler. |
| `yolo26:orchestration` | Wasm preprocessing, graph invocation, and postprocessing flow. |
| `yolo26:preprocess` | Native OpenCV image decoder and resize provider. |

Select the package as the application entrypoint:

```toml
[entrypoint]
package = "yolo26"
```

The root application supplies the checkpoint and chooses the backend and weight
format policies. The standalone `synps.toml` in this repository selects CUDA
and F32 so it can also be built and tested directly.

## Input and output

The public input is an encoded image buffer. The native preprocessing stage
decodes and resizes the image before graph execution.

The artifact emits one output value containing a list of detections. Each
detection contains:

| Field | Meaning |
|---|---|
| `label` | Detected class name |
| `confidence` | Confidence score |
| `x1`, `y1` | Top-left corner of the bounding box |
| `x2`, `y2` | Bottom-right corner of the bounding box |

Detections below the model's confidence threshold are omitted and the remaining
items are ordered from highest to lowest confidence. The Runtime selects the
typed return conversion automatically and delivers the list in one callback
invocation. This is a single-output workload even though it uses the same
stream callback contract as generative models.

Load the finished artifact through `Mln::Runtime`, create a Session, and invoke
the public `yolo26` entrypoint with the encoded image Buffer.

## Documentation

Full documentation lives at **[https://useplx.github.io/packages/yolo26/](https://useplx.github.io/packages/yolo26/)**.

- [Package guide](https://useplx.github.io/packages/yolo26/)

## License

This repository follows the upstream
[GNU Affero General Public License v3.0](LICENSE). Ultralytics also offers a
separate Enterprise License for uses that do not follow AGPL-3.0 requirements.

Users must review the terms attached to model weights, datasets, names, and
trademarks separately.
