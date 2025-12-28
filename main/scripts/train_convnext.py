
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

if cfg.num_classes != 6: raise ValueError("Your dataloader enforces exactly 6 classes (CLASS_ORDER).")

seed_everything(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# create run folder inside results/
run_name = datetime.now().strftime("fer_cnnvanilla_%Y%m%d_%H%M%S")
results_root = cfg.results_root.resolve()
out_dir = results_root / run_name
out_dir.mkdir(parents=True, exist_ok=True)

# save config + mappings
(out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2, default=str))
(out_dir / "class_to_idx.json").write_text(json.dumps(CLASS_TO_IDX, indent=2))
(out_dir / "class_order.json").write_text(json.dumps(CLASS_ORDER, indent=2))

# dataloaders
dls = build_dataloaders(cfg.images_root, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
train_loader, val_loader, test_loader = dls.train, dls.val, dls.test

# import weights(for transfer learning)
weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
model = convnext_tiny(weights=weights)

# replace classifier
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features,cfg.num_classes)

# loss
if cfg.use_class_weights:
    w = compute_class_weights(train_loader, cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    (out_dir / "class_weights.json").write_text(json.dumps(w.detach().cpu().tolist(), indent=2))
else:
    criterion = nn.CrossEntropyLoss()

# freezing the weightsa(unfreeze )
for para in model.features.parameters():
    para.requires_grad = False

model = model.to(device)

optimizer = AdamW(filter(lambda p : p.requires_grad,model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)

scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.min_lr)

# track best by macro-F1
best_val_f1 = -1.0
best_epoch = -1
best_state = None
patience_left = cfg.early_stop_patience

for epoch in range(1,cfg.epochs +1):
    train_loss = train_one_epoch(model,train_loader,criterion, optimizer, device, scaler, use_amp,cfg.grad_clip)
    scheduler.step()
    print(f"Epoch{epoch}")
    val_loss, val_metrics, val_cm = evaluate(model, val_loader, criterion, device, cfg.num_classes)
    val_f1 = float(val_metrics.get("f1_macro", 0.0))
    val_acc = float(val_metrics.get("accuracy", 0.0))

    print(
        f"Epoch {epoch:02d}/{cfg.epochs} | "
        f"train loss {train_loss:.4f} | "
        f"val loss {val_loss:.4f} | "
        f"val acc {val_acc:.4f} | "
        f"val f1_macro {val_f1:.4f}"
        )
    # store epoch artifacts
    np.save(out_dir / f"val_cm_epoch_{epoch:03d}.npy", val_cm)
    (out_dir / f"val_metrics_epoch_{epoch:03d}.json").write_text(json.dumps(val_metrics, indent=2))

    # best checkpoint
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        patience_left = cfg.early_stop_patience

        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        save_best_checkpoint(out_dir, epoch, model, optimizer, best_val_f1, cfg)
    else:
        patience_left -= 1

    if cfg.early_stop_patience > 0 and patience_left <= 0:
        print(f"Early stopping. Best epoch: {best_epoch}, best val f1_macro: {best_val_f1:.4f}")
        break

    # restore best weights for test + exports
    if best_state is not None:
        model.load_state_dict(best_state)

    # test evaluation
    test_loss, test_metrics, test_cm = evaluate(model, test_loader, criterion, device, cfg.num_classes)
    print(
        f"TEST | loss {test_loss:.4f} | "
        f"acc {float(test_metrics.get('accuracy', 0.0)):.4f} | "
        f"f1_macro {float(test_metrics.get('f1_macro', 0.0)):.4f}"
    )
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    np.save(out_dir / "test_cm.npy", test_cm)

    # preview images OUTSIDE results/
    preview_dir = results_root.parent / f"preview_{run_name}"
    split = cfg.preview_split.lower().strip()
    preview_loader = train_loader if split == "train" else val_loader if split == "val" else test_loader

    save_preview_images(
        model=model,
        loader=preview_loader,
        device=device,
        out_dir=preview_dir,
        idx_to_class_map=idx_to_class(),
        n=cfg.preview_n,
    )
    print(f"Preview images saved to: {preview_dir}")

    # frozen model export
    export_frozen_model(out_dir, model, cfg.image_size)
    print(f"Saved run to: {out_dir}")
    print(f"- best checkpoint: {out_dir / 'best.ckpt.pt'}")
    print(f"- frozen model:    {out_dir / 'model_frozen.ts'}")
    print(f"- state_dict only: {out_dir / 'model_state_dict.pt'}")

#------------------
# unfreeze backbone
#------------------
for p in model.features[-2:].parameters():
    p.requires_grad = True

optimizer = AdamW(model.parameters(), lr=cfg.lr * 0.1, weight_decay=cfg.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.min_lr)

#reset
best_val_f1 = -1.0
best_epoch = -1
best_state = None
patience_left = cfg.early_stop_patience

for epoch in range(1,cfg.epochs +1):
    train_loss = train_one_epoch(model,train_loader,criterion, optimizer, device, scaler, use_amp,cfg.grad_clip)
    scheduler.step()

    print(f"Epoch{epoch}")
    val_loss, val_metrics, val_cm = evaluate(model, val_loader, criterion, device, cfg.num_classes)
    val_f1 = float(val_metrics.get("f1_macro", 0.0))
    val_acc = float(val_metrics.get("accuracy", 0.0))

    print(
        f"Epoch {epoch:02d}/{cfg.epochs} | "
        f"train loss {train_loss:.4f} | "
        f"val loss {val_loss:.4f} | "
        f"val acc {val_acc:.4f} | "
        f"val f1_macro {val_f1:.4f}"
        )
    # store epoch artifacts
    np.save(out_dir / f"val_cm_epoch_{epoch:03d}.npy", val_cm)
    (out_dir / f"val_metrics_epoch_{epoch:03d}.json").write_text(json.dumps(val_metrics, indent=2))

    # best checkpoint
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        patience_left = cfg.early_stop_patience

        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        save_best_checkpoint(out_dir, epoch, model, optimizer, best_val_f1, cfg)
    else:
        patience_left -= 1

    if cfg.early_stop_patience > 0 and patience_left <= 0:
        print(f"Early stopping. Best epoch: {best_epoch}, best val f1_macro: {best_val_f1:.4f}")
        break

    # restore best weights for test + exports
    if best_state is not None:
        model.load_state_dict(best_state)

    # test evaluation
    test_loss, test_metrics, test_cm = evaluate(model, test_loader, criterion, device, cfg.num_classes)
    print(
        f"TEST | loss {test_loss:.4f} | "
        f"acc {float(test_metrics.get('accuracy', 0.0)):.4f} | "
        f"f1_macro {float(test_metrics.get('f1_macro', 0.0)):.4f}"
    )
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    np.save(out_dir / "test_cm.npy", test_cm)

    # preview images OUTSIDE results/
    preview_dir = results_root.parent / f"preview_{run_name}"
    split = cfg.preview_split.lower().strip()
    preview_loader = train_loader if split == "train" else val_loader if split == "val" else test_loader

    save_preview_images(
        model=model,
        loader=preview_loader,
        device=device,
        out_dir=preview_dir,
        idx_to_class_map=idx_to_class(),
        n=cfg.preview_n,
    )
    print(f"Preview images saved to: {preview_dir}")

    # frozen model export
    export_frozen_model(out_dir, model, cfg.image_size)
    print(f"Saved run to: {out_dir}")
    print(f"- best checkpoint: {out_dir / 'best.ckpt.pt'}")
    print(f"- frozen model:    {out_dir / 'model_frozen.ts'}")
    print(f"- state_dict only: {out_dir / 'model_state_dict.pt'}")
