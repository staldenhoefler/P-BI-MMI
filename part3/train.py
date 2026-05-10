import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from dataset import QuizDataModule
from model import QuizModel

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"Training with target variable: {config['data']['target_variable']}")

    datamodule = QuizDataModule(config)

    datamodule.setup('fit')
    
    model = QuizModel(input_dim=datamodule.input_dim, config=config)
    
    # Initialize Logger
    wandb_logger = None
    if 'logging' in config:
        wandb_logger = WandbLogger(
            project=config['logging'].get('project', 'quiz_prediction'),
            name=config['logging'].get('name', 'run'),
            config=config
        )

    # 4. Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        logger=wandb_logger,
    )
    
    # 5. Train Model
    trainer.fit(model, datamodule=datamodule)
    
    # 6. Test Model
    trainer.test(model, datamodule=datamodule)

if __name__ == '__main__':
    main()
