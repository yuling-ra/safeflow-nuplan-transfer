#!/usr/bin/env bash
python -m src.train \
  --config configs/config.yaml \
  --override "model.attn.enable_cbam=true,train.optimizer.lr=5e-4"
