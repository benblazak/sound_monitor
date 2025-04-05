# sound monitor

monitor sounds and record dog barks

## hardware

- Raspberry Pi 4b
- MiniDSP UMA-8 microphone array

## software

### configure

see [sound_monitor/config.py](./sound_monitor/config.py)

### install

```bash
./pre-install.sh
python -m pip install .
```

### run

```bash
sound-monitor
```

### develop

install [mise](https://mise.jdx.dev/getting-started.html)

```bash
./pre-install.sh
mise install  # install tools
mise exec -- python -m pip install --editable ".[dev]"
```

see [mise tasks](./.mise.toml)

## references

- [yamnet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
  - [model](https://www.kaggle.com/models/google/yamnet)
- [odas](https://github.com/introlab/odas)
  - [configuration](https://github.com/introlab/odas/wiki/configuration)
    - [minidsp uma8](https://github.com/introlab/odas/blob/master/config/odaslive/minidsp.cfg)
