#!/bin/bash
# patches img2img-turbo source files to use MPS instead of CUDA for Apple Silicon
sed -i '' 's/\.cuda()/.to("mps")/g' img2img-turbo/src/pix2pix_turbo.py
sed -i '' 's/\.cuda()/.to("mps")/g' img2img-turbo/src/model.py
sed -i '' 's/device="cuda"/device="mps"/g' img2img-turbo/src/model.py
sed -i '' 's/unet.to("cuda")/unet.to("mps")/g' img2img-turbo/src/pix2pix_turbo.py
sed -i '' 's/vae.to("cuda")/vae.to("mps")/g' img2img-turbo/src/pix2pix_turbo.py
sed -i '' 's/device="cuda")/device="mps")/g' img2img-turbo/src/pix2pix_turbo.py
echo "MPS patches applied successfully"