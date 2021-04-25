# Pose estimation

## AlphaPose

[github](https://github.com/MVIG-SJTU/AlphaPose)

Activate env:
```source ~/Projects/envs/alpha-pose/bin/activate```

```python3 scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/ --save_img```

```python scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --indir examples/demo/ --save_img```

You can specify outdir:
```--outdir```
