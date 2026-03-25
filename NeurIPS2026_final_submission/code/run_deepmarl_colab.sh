#!/bin/bash
set -e
echo '[1/5] Cloning Repository...'
git clone https://github.com/Yesol-Pilot/EthicaAI.git ethicaai_colab || true
cd ethicaai_colab/NeurIPS2026_final_submission/code

echo '[2/5] Setting up Environment...'
pip install pettingzoo higher > /dev/null 2>&1

echo '[3/5] Running Deep MARL GPU Experiments (This will take hours)...'
cd scripts
python pytorch_reinforce_nash_trap.py
python maccl_pytorch.py
python lio_pytorch.py
python rnd_ppo.py

echo '[4/5] Aggregating Results...'
cd ..
mkdir -p colab_deepmarl_results
cp -r outputs/deep_models colab_deepmarl_results/ || true
cp -r outputs/pytorch_reinforce colab_deepmarl_results/ || true
zip -r colab_deepmarl_results.zip colab_deepmarl_results

echo '[5/5] Success! Please download colab_deepmarl_results.zip'
