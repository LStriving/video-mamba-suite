#!bin/bash
echo "Stage 2: Swallow"
for i in {1..7}
do
    python train2stage.py ./configs/2stage/parrallel/mamba_swallow_i3d_secondstage_$i.yaml --output novswg
done