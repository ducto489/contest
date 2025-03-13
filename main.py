import argparse
import os
from src.trainer import Trainer
from src.inference import Inferencer
from src.utils.platform_config import load_config, setup_environment

def main():
    parser = argparse.ArgumentParser(description='Object Detection Pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True,
                       help='Mode to run: train or inference')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file (required for inference)')
    parser.add_argument('--data_dir', type=str, help='Path to data directory (optional, overrides config)')
    
    args = parser.parse_args()
    
    # Load and update configuration based on platform
    config, platform = load_config(args.config)
    
    # Setup platform-specific environment
    setup_environment(platform)
    
    # Override data directory if provided
    if args.data_dir:
        config['data']['train_path'] = os.path.join(args.data_dir, 'train')
        config['data']['test_path'] = os.path.join(args.data_dir, 'test')
    
    if args.mode == 'train':
        trainer = Trainer(config)
        trainer.train()
    else:
        if args.checkpoint is None:
            raise ValueError("Checkpoint path is required for inference mode")
        inferencer = Inferencer(config, args.checkpoint)
        inferencer.run_inference()

if __name__ == '__main__':
    main() 