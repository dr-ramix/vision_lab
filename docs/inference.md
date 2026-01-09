// ============================================================
// FER Inference & Model Hub
// ============================================================
//
// This repository implements a modular system for training,
// validating, and running inference with multiple
// Facial Emotion Recognition (FER) model architectures.
//
// Core principles:
//   - Model architectures live in code (Git)
//   - Pretrained weights live outside Git (Hugging Face)
//   - Inference always reconstructs the exact training setup
//   - One unified API for all models
//
// The system is designed for research, reproducibility,
// and collaboration across machines and environments.
// ============================================================



// ============================================================
// WHAT IS CODE vs WHAT IS DATA
// ============================================================
//
// CODE (tracked by Git):
//   - All Python source files
//   - Model architectures
//   - Registries
//   - Inference logic
//   - Utility scripts
//
// DATA / WEIGHTS (NOT tracked by Git):
//   - model_state_dict.pt
//   - model.safetensors
//   - config.json next to weights
//
// Weights are hosted externally and loaded on demand.
// ============================================================



// ============================================================
// QUICK START
// ============================================================
//
// Minimal inference usage:
//
//   from fer.inference.models import ResNet50
//   model = ResNet50().load()
//
// This:
//   - finds the correct pretrained weights
//   - reconstructs the architecture
//   - loads the checkpoint
//   - returns a ready-to-use torch.nn.Module
//
// ============================================================



// ============================================================
// REPOSITORY STRUCTURE (IMPORTANT PARTS)
// ============================================================
//
// main/src/fer/models/
//   - Training architectures only
//   - NO pretrained weights here
//
// main/src/fer/models/registry.py
//   - Central registry that defines how models are constructed
//   - Used by BOTH training and inference
//
// main/src/fer/inference/
//   - Inference subsystem
//   - User-facing API
//
// main/src/fer/inference/models.py
//   - Defines importable inference classes
//   - Example: ResNet50, ConvNeXtBase, EmoCatNetsTiny
//
// main/src/fer/inference/hub.py
//   - Handles WHERE weights come from:
//       * local project folder
//       * project-level download
//       * Hugging Face cache
//
// main/src/fer/inference/weights/
//   - Optional local storage for pretrained weights
//   - This folder is NOT tracked by Git
//
// main/scripts/
//   - Small, explicit utility scripts
//   - Downloading, verification, testing
// ============================================================



// ============================================================
// PRETRAINED WEIGHTS (OUTSIDE GIT)
// ============================================================
//
// All pretrained weights are hosted on Hugging Face:
//
//   https://huggingface.co/drRamix/EMO_NETS_LMU
//
// Each model has ONE folder (weights_id), e.g.:
//
//   resnet50/
//     model_state_dict.pt
//     config.json
//     README.md
//
// The folder name (weights_id):
//   - matches the inference model definition
//   - is used by scripts
//   - is used by the hub resolver
//
// Weights can be:
//   - copied manually into the project
//   - downloaded via scripts
//   - loaded directly from the HF cache
// ============================================================



// ============================================================
// config.json (CRITICAL FILE)
// ============================================================
//
// Each weights folder contains a config.json describing
// how the model was trained.
//
// Example fields:
//   - num_classes
//   - in_channels
//   - transfer
//   - class_names
//
// Inference ALWAYS reads this file to:
//   - build the correct classifier head
//   - avoid silent shape mismatches
//   - guarantee training/inference consistency
// ============================================================



// ============================================================
// SCRIPTS (main/scripts/)
// ============================================================
//
// download_weights.py
//   - Downloads pretrained weights from Hugging Face
//   - Places them into fer/inference/weights/
//
//   Examples:
//     python main/scripts/download_weights.py --all
//     python main/scripts/download_weights.py --model resnet50
//
//
// verify_weights.py
//   - End-to-end validation of each model:
//       * weights files exist and are non-empty
//       * architecture matches registry
//       * classifier head size is correct
//       * checkpoint loads with no missing keys
//
//   Example:
//     python main/scripts/verify_weights.py
//
//
//
// test_import_model.py
//   - Stronger sanity check
//   - Runs a forward pass on random input
//   - Confirms output shape and numerical stability
//
//   Example:
//     python main/scripts/test_import_model.py
// ============================================================



// ============================================================
// HOW INFERENCE WORKS (END-TO-END PIPELINE)
// ============================================================
//
// 1) User imports a model class from fer.inference.models
//
// 2) The inference wrapper resolves the weights folder
//    using weights_id and the hub resolver
//
// 3) config.json is read to recover training parameters
//
// 4) The architecture is constructed via
//    fer.models.registry.make_model()
//
// 5) model_state_dict.pt is loaded into the model
//
// 6) The model is moved to the requested device
//    and returned in evaluation mode
//
// This pipeline is identical for ALL models.
// ============================================================



// ============================================================
// USING A MODEL (GPU / CPU)
// ============================================================
//
// Example:
//
//   import torch
//   from fer.inference.models import ConvNeXtBase
//
//   device = "cuda" if torch.cuda.is_available() else "cpu"
//   model = ConvNeXtBase().load(device=device)
//   model.eval()
//
// Forward pass:
//
//   x = torch.randn(2, 3, 64, 64, device=device)
//
//   with torch.no_grad():
//       logits = model(x)
//       probs  = torch.softmax(logits, dim=1)
//       pred   = probs.argmax(dim=1)
//
// ============================================================



// ============================================================
// ADDING A NEW MODEL (CHECKLIST)
// ============================================================
//
// 1) Implement architecture in:
//      main/src/fer/models/
//
// 2) Register architecture in:
//      main/src/fer/models/registry.py
//
// 3) Train model and save:
//      model_state_dict.pt
//
// 4) Create config.json describing training setup
//
// 5) Upload weights folder to Hugging Face
//
// 6) Register inference wrapper in:
//      main/src/fer/inference/models.py
//
// After this, the model becomes available via:
//
//   from fer.inference.models import MyNewModel
//   model = MyNewModel().load()
//
// ============================================================



// ============================================================
// CONVENTIONS / NOTES
// ============================================================
//
// - Default setup uses 6-class FER
// - Large binaries are intentionally excluded from Git
// - config.json is mandatory for safety
// - Always run verify_weights.py after adding new weights
//
// ============================================================
