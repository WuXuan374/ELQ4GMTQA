# ELQ4GMTQA
Tailored ELQ for entity linking of [GMT-KBQA](https://github.com/HXX97/GMT-KBQA)
## Easy Start
- Create an conda environment accroding to configuration file `blink37.yml`:
```bash
conda env create -f blink37.yml
```
- Activate environment:
```bash
conda activate blink37
```
- Start ELQ service, the default port is `5688`:
```bash
CUDA_VISIBLE_DEVICES=3 python elq_service.py
```