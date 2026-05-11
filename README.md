# Self-Supervised Feature Representation Learning for CLDN18.2 Detection in Gastric Cancer iSyntax Whole-Slide Images

This project leverages Philips iSyntax whole-slide imaging (WSI) to perform high-resolution morphological analysis of gastric cancer tissue. Utilizing the Philips Pathology SDK and PixelEngine, we have developed a pipeline that extracts structured image patches from large-scale iSyntax slides, ensuring the preservation of critical spatial and contextual information. This framework enables the scalable processing of gigapixel pathology data, transforming massive raw files into a structured foundation for downstream computational analysis.

A distinctive feature of this study is the integration of VYLOY-stained (CLDN18.2) histopathology slides, which provide enhanced molecular and structural contrast for the precise identification of tumor-specific patterns. By synchronizing systematic patch extraction with VYLOY-specific features, the project captures subtle morphological variations indicative of gastric cancer progression that are often elusive in traditional workflows. This dual approach facilitates both high-fidelity qualitative interpretation and robust quantitative modeling.

The resulting dataset powers machine learning workflows designed to discriminate between malignant and benign regions, fostering the development of automated diagnostic support systems. By bridging the gap between proprietary iSyntax data acquisition and advanced vision models, this project advances the standards for classification accuracy, early detection, and deep research insights into the gastric cancer landscape.

# Objectives
1. Objective Quantification and Reproducibility: 
The integration of DINOv2 into digital pathology addresses the inherent subjectivity and variability associated with manual visual assessment. While a pathologist may estimate the CLDN18.2 expression percentage, human observation is often prone to inter-observer bias, especially near critical clinical thresholds. By leveraging the high-dimensional feature extraction capabilities of DINOv2, we can implement a standardized computational framework that precisely quantifies staining intensity and spatial distribution at the pixel level. This ensures highly reproducible results that are essential for making consistent clinical decisions regarding VYLOY eligibility.

2. Discovery of Sub-visual Morphological Features: 
Beyond simple color recognition, DINOv2’s self-supervised learning architecture excels at identifying complex, sub-visual patterns that elude human perception. The model generates rich, dense embeddings that capture intricate morphological nuances, such as subtle alterations in nuclear architecture, cellular orientation, and the microenvironmental relationship between tumor cells and the surrounding stroma. These latent features allow for a deeper characterization of the "Positive" slides, potentially uncovering structural biomarkers that correlate with treatment response, which cannot be identified by traditional staining analysis alone.

3. Enhancing Diagnostic Efficiency and Workflow: 
Processing high-resolution iSyntax files is computationally and labor-intensively demanding due to the massive scale of Whole Slide Images (WSI). DINOv2 facilitates a significant optimization of the diagnostic workflow by acting as an intelligent triage system. The model can rapidly scan thousands of image patches to pinpoint specific Regions of Interest (ROIs) where CLDN18.2 expression is most prominent or clinically significant. By automating the screening process and filtering out non-informative areas, the system allows pathologists to focus their expertise on the most critical diagnostic regions, thereby drastically reducing turnaround time and increasing overall throughput.

# Pipeline
1. convert_isyntax_to_zarr.py
   
   * Input:  ./slides/*.isyntax
   * Output: ./zarr/[Slide_ID].zarr
           zarr_conversion_logs.csv

3. zarr_clean_manifest.py
   
   * Input:  ./zarr/[Slide_ID].zarr
   * Output: clean_manifest.csv
           QC images for checking main tissue/noise detection

5. dinov2_final.py
   
   * Input:  clean_manifest.csv
   * Output: DINOv2 training/features using only clean 518×518 Level 0 patches

# Installation on Ubuntu
* Ubuntu 20 on Kakao Cloud with 8 B200 GPUs.
* 200 iSyntax files with VYLOY-Based Features

* sudo sed -i 's|http://ftp.daum.net/ubuntu|http://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list
* sudo add-apt-repository ppa:deadsnakes/ppa
* sudo apt update
* sudo apt install python3.8 python3.8-venv python3.8-distutils
* sudo apt install libgles2 libegl1
* sudo apt install liblcms2-2
* sudo apt install libtinyxml2.6.2v5 libgles2-mesa libegl1-mesa
* sudo dpkg -i --ignore-depends=python3 deb_files/philips-pathologysdk-python3-softwarerendercontext_5.1.0-1_all.deb
* sudo dpkg -i --ignore-depends=python3 deb_files/philips-pathologysdk-pixelengine_5.1.0-1_amd64.deb 
* sudo dpkg -i --ignore-depends=python3 deb_files/philips-pathologysdk-python3-pixelengine_5.1.0-1_all.deb
* sudo dpkg -i --ignore-depends=python3 deb_files/philips-pathologysdk-softwarerenderer_5.1.0-1_amd64.deb 
* #PYTHONPATH=$HOME/.local/lib/python3.8/site-packages:/usr/lib/python3/dist-packages
* python3.8 -c "import pixelengine; print('pixelengine OK')"
* python3.8 -c "import pixelengine; print(pixelengine.__file__)"
* #export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
* #export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
* python3.8 -m pip install opencv-python-headless
* sudo apt-get install -y libgl1-mesa-glx
* sudo apt --fix-broken install
 
# Execution Steps
- pyton3.8 convert_isyntax_to_zarr.py

- pyton3.8 zarr_clean_manifest.py

- python3.11 -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  dinov2_final.py \
  --manifest ./clean_manifest.csv \
  --output-dir ./dinov2_features \
  --mode extract \
  --backbone dinov2_vitb14 \
  --dinov2-repo /home/jovyan/dinov2 \
  --tile-size 518 \
  --batch-size 32 \
  --num-workers 0


*Sponsored by GensionBio*
