# sound monitor

monitor sounds and record animal sounds

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

## TODO

- [ ] when we switch to a new file around midnight
  - [ ] grep for warnings and errors and such in the log, and email them
- [ ] recording in stereo might not turn out to be useful. if not we can switch to mono
- [ ] for storing yamnet data
  - [ ] claude.ai "Efficient Storage for Real-Time Score Data"
  - [ ] keep raw data, quantize (to 3 decimal places) and compress using numpy, store as binary in sqlite
    - [ ] can maybe write a view in the db for decoding, so i can see it in a db viewer? not sure about this
  - [ ] daily
    - [ ] take max of 'animal', 95th percentile of other, over 5s windows
  - [ ] weekly
    - [ ] take max of 'animal', 95th percentile of other, over larger windows, starting from the 5s window data
