FROM mambaorg/micromamba:0.27.0

USER root

WORKDIR /app/

COPY --chown=$MAMBA_USER:$MAMBA_USER environment-docker.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

RUN mkdir -p models
RUN mkdir -p /dicom
COPY models/catboost.percent_phase_field_of_view:sar:series_description.pkl models

COPY src src
COPY predict-docker.sh predict.sh
RUN chmod +x predict.sh

ARG MAMBA_DOCKERFILE_ACTIVATE=1

ENTRYPOINT ["./predict.sh"]