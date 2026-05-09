# AI-Driven Gastric Cancer Detection from Philips iSyntax Slides with VYLOY-Based Histological Features

This project focuses on leveraging Philips iSyntax whole-slide imaging data to enable high-resolution analysis of tissue morphology in the context of gastric cancer. Using the Philips Pathology SDK and PixelEngine, the pipeline extracts structured image patches from large-scale iSyntax slides while preserving spatial and contextual information. These patches serve as the foundation for downstream computational analysis, allowing scalable processing of gigapixel pathology data.

A key component of the study is the integration of VYLOY-stained histopathology slides, which enhance cellular and structural contrast for more precise identification of tissue patterns. By combining VYLOY staining with systematic patch extraction from iSyntax files, the project aims to capture subtle morphological variations associated with gastric cancer progression. This approach supports both qualitative pathology interpretation and quantitative modeling.

The extracted data are then utilized in machine learning workflows to distinguish between cancerous and non-cancerous regions, enabling automated or semi-automated diagnostic support. By bridging Philips iSyntax-based data acquisition with advanced analytical techniques, the project contributes toward improving early detection, classification accuracy, and research insights in gastric cancer pathology.

# System Installation
sudo sed -i 's|http://ftp.daum.net/ubuntu|http://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-distutils
sudo apt install libgles2 libegl1
sudo apt install liblcms2-2
sudo apt install libtinyxml2.6.2v5 libgles2-mesa libegl1-mesa

sudo dpkg -i --ignore-depends=python3 deb_files/philips-pathologysdk-python3-softwarerendercontext_5.1.0-1_all.deb
sudo dpkg -i --ignore-depends=python3 deb_files/philips-pathologysdk-pixelengine_5.1.0-1_amd64.deb 
sudo dpkg -i --ignore-depends=python3 deb_files/philips-pathologysdk-python3-pixelengine_5.1.0-1_all.deb
sudo dpkg -i --ignore-depends=python3 deb_files/philips-pathologysdk-softwarerenderer_5.1.0-1_amd64.deb 

##PYTHONPATH=$HOME/.local/lib/python3.8/site-packages:/usr/lib/python3/dist-packages
python3.8 -c "import pixelengine; print('pixelengine OK')"
python3.8 -c "import pixelengine; print(pixelengine.__file__)"

##export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
##export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH

python3.8 -m pip install opencv-python-headless
sudo apt-get install -y libgl1-mesa-glx
sudo apt --fix-broken install
