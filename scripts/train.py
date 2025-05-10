import os
import json
import subprocess

# Cargar configuración
with open("configs/lora_config.json", "r") as f:
    config = json.load(f)

# Construir el comando
cmd = [
    "accelerate", "launch",
    "--mixed_precision=fp16",
    "--multi_gpu", "train_network.py",
    f"--pretrained_model_name_or_path={config['pretrained_model_name_or_path']}",
    f"--train_data_dir={config['train_data_dir']}",
    f"--output_dir={config['output_dir']}",
    f"--resolution={config['resolution']}",
    f"--train_batch_size={config['train_batch_size']}",
    f"--max_train_steps={config['max_train_steps']}",
    f"--learning_rate={config['learning_rate']}",
    f"--lr_scheduler={config['lr_scheduler']}",
    f"--network_dim={config['network_dim']}",
    f"--network_alpha={config['network_alpha']}",
    "--network_module=networks.lora",
    "--output_name=escudo_lora"
]

print("Ejecutando entrenamiento...")
subprocess.run(cmd)
