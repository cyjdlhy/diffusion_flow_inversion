# CUDA_VISIBLE_DEVICES=0 python -m debugpy \
#        --listen 0.0.0.0:4681 \
#        --wait-for-client \
#        main.py


CUDA_VISIBLE_DEVICES=0 python -m debugpy \
       --listen 0.0.0.0:4681 \
       --wait-for-client \
       comparison_Flux.py

