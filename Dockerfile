FROM ubuntu:24.04

# aptの更新など
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y wget

# CUDAのインストール
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
# ↓Ubuntu23.04以降でCUDAのインストールに必要なコマンド
RUN echo "deb http://archive.ubuntu.com/ubuntu/ jammy universe" >> /etc/apt/sources.list.d/jammy.list
RUN echo "Package: *\n\
    Pin: release n=jammy\n\
    Pin-Priority: -10\n\n\
    Package: libtinfo5\n\
    Pin: release n=jammy\n\
    Pin-Priority: 990" >> /etc/apt/preferences.d/pin-jammy
# ↑Ubuntu23.04以降でCUDAのインストールに必要なコマンド
RUN apt-get update
RUN apt-get -y install cuda-toolkit-12-4
RUN rm cuda-keyring_1.1-1_all.deb
RUN echo "export PATH=/usr/local/cuda-12.4/bin\${PATH:+:\${PATH}}" >> /root/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\n"  >> /root/.bashrc

# .bashrcの編集
RUN sed -i 1iforce_color_prompt=yes /root/.bashrc
RUN echo alias la="'ls -lah'\n" >> /root/.bashrc

# pyenvのインストール
RUN apt-get install -y curl git build-essential libssl-dev libffi-dev libncurses5-dev zlib1g zlib1g-dev libreadline-dev libbz2-dev libsqlite3-dev make gcc liblzma-dev python-tk tk-dev
RUN curl https://pyenv.run | bash
RUN echo "export PYENV_ROOT=\"\$HOME/.pyenv\""  >> /root/.bashrc
RUN echo "export PATH=\"\$PYENV_ROOT/bin:\$PATH\""  >> /root/.bashrc
RUN echo "eval \"\$(pyenv init -)\""  >> /root/.bashrc
RUN echo "eval \"\$(pyenv virtualenv-init -)\""  >> /root/.bashrc

RUN apt-get install -y htop vim