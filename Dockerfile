FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Need Wget to install Miniconda.
RUN  apt-get update && apt-get install -y wget 

# Install Miniconda.
RUN mkdir -p /miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh
RUN bash /miniconda3/miniconda.sh -b -u -p /miniconda3
RUN rm /miniconda3/miniconda.sh

ENV PATH="/miniconda3/bin:${PATH}"

WORKDIR /NetPress
COPY --chown=user . .

# Build dependencies.
RUN apt-get install -y build-essential

# Create conda environments.
RUN conda tos accept
RUN conda env create -f environment_ai_gym.yml
RUN conda env create -f environment_mininet.yml

# Install Flash Attention separately since it requires PyTorch at build time...
RUN conda init --all
RUN conda run -n ai_gym_env pip install flash-attn==2.7.4.post1

# Make sudo dummy replacement, so we don't weaken docker security. Mininet (app-route) uses sudo as well.
RUN echo "#!/bin/bash\n\$@" > /usr/bin/sudo
RUN chmod +x /usr/bin/sudo

RUN rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]