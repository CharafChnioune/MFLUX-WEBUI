import os
import sys
import json
import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

# Voeg de backend directory toe aan het pad voor imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Imports die alleen tijdens runtime nodig zijn, niet tijdens type-checking
if not TYPE_CHECKING:
    try:
        import mlx.core.random as random
    except ImportError:
        random = None

    # Importeer mflux klassen
    from mflux import Flux1
    from mflux_compat import Config, ModelConfig, RuntimeConfig
    from mflux.dreambooth.dataset.dataset import Dataset
    from mflux.dreambooth.dataset.iterator import Iterator
    from mflux.dreambooth.lora_layers.lora_layers import LoRALayers
    from mflux.dreambooth.optimization.optimizer import Optimizer
    from mflux.dreambooth.state.training_spec import TrainingSpec
    from mflux.dreambooth.state.training_state import TrainingState
    from mflux.dreambooth.statistics.statistics import Statistics
    from mflux.dreambooth.dreambooth import DreamBooth
    from mflux.error.exceptions import StopTrainingException
    from mflux.tokenizer.tokenizer_handler import TokenizerHandler

    # Importeer custom model config
    from model_manager import CustomModelConfig, MODELS
else:
    # Type hints voor de editor, deze worden alleen gebruikt tijdens type-checking
    class Flux1:
        pass
    
    class ModelConfig:
        def __init__(
            self,
            alias: str | None,
            model_name: str,
            base_model: str | None,
            num_train_steps: int,
            max_sequence_length: int,
            supports_guidance: bool,
        ):
            self.alias = alias
            self.model_name = model_name
            self.base_model = base_model
            self.num_train_steps = num_train_steps
            self.max_sequence_length = max_sequence_length
            self.supports_guidance = supports_guidance
    
    class Config:
        pass
    
    class RuntimeConfig:
        pass
    
    class TrainingSpec:
        @staticmethod
        def resolve(config_path: Optional[str], checkpoint_path: Optional[str]):
            return None
        
        seed: int
        steps: int
        width: int
        height: int
        guidance: float
        quantize: Optional[int]
        examples: dict
    
    class TrainingState:
        pass
    
    class Dataset:
        @staticmethod
        def prepare_dataset(*args, **kwargs):
            return None
    
    class Iterator:
        @staticmethod
        def from_spec(*args, **kwargs):
            return None
    
    class LoRALayers:
        @staticmethod
        def from_spec(*args, **kwargs):
            return None
    
    class Optimizer:
        @staticmethod
        def from_spec(*args, **kwargs):
            return None
    
    class Statistics:
        @staticmethod
        def from_spec(*args, **kwargs):
            return None
    
    class DreamBooth:
        @staticmethod
        def train(*args, **kwargs):
            pass
    
    class CustomModelConfig:
        model_name: str
        alias: str
        num_train_steps: int
        max_sequence_length: int
        base_arch: str
    
    class StopTrainingException(Exception):
        pass
    
    # Type hint voor de model_manager module
    MODELS = {
        "dev": CustomModelConfig(),
        "schnell": CustomModelConfig(),
        "dev-4-bit": CustomModelConfig(),
    }

class CustomDreamBoothInitializer:
    @staticmethod
    def initialize(
        config_path: str | None,
        checkpoint_path: str | None,
        model_type: str = "dev",
        local_model_path: str | None = None,
        quantize: int | None = None,
    ):
        """
        Initialiseer een Dreambooth training sessie met een custom model.
        
        Args:
            config_path: Pad naar het configuratie bestand
            checkpoint_path: Pad naar een checkpoint bestand om training te hervatten
            model_type: Type model (dev of schnell)
            local_model_path: Pad naar het lokale model
            quantize: Kwantisatie bits (3, 4, 6, 8)
        """
        # Laad de training specificatie
        training_spec = TrainingSpec.resolve(
            config_path=config_path,
            checkpoint_path=checkpoint_path
        )
        
        # Set global random seed voor deterministische training
        if not TYPE_CHECKING:
            if 'random' in globals() and random is not None:
                random.seed(training_spec.seed)
            else:
                print("Waarschuwing: mlx.core.random niet beschikbaar, random seed niet ingesteld")
        
        # Overschrijf de model type met onze custom model
        if model_type not in MODELS:
            print(f"Warning: Model type {model_type} niet gevonden, gebruik 'dev' als fallback")
            model_type = "dev"
            
        # Voeg kwantisatie toe aan model type indien nodig
        if quantize is not None and f"{model_type}-{quantize}-bit" in MODELS:
            model_type = f"{model_type}-{quantize}-bit"
        
        print(f"Initialiseren met model: {model_type}")
        
        # Gebruik de CustomModelConfig uit model_manager
        custom_config = MODELS[model_type]
        model_config = ModelConfig(
            alias=custom_config.alias,
            model_name=custom_config.model_name,
            base_model=None,  # Base model is None in de oorspronkelijke implementatie voor dev/schnell
            num_train_steps=custom_config.num_train_steps,
            max_sequence_length=custom_config.max_sequence_length,
            supports_guidance=custom_config.supports_guidance
        )
        
        # Initialiseer de Flux1 klasse met onze eigen implementatie
        flux = CustomFlux1(
            model_config=model_config,
            quantize=training_spec.quantize,
            local_path=local_model_path,
        )
        
        runtime_config = RuntimeConfig(
            model_config=model_config,
            config=Config(
                num_inference_steps=training_spec.steps,
                width=training_spec.width,
                height=training_spec.height,
                guidance=training_spec.guidance,
            ),
        )

        # Maak de optimizer
        optimizer = Optimizer.from_spec(training_spec)

        # Maak de LoRA layers door ze te matchen tegen de Flux layers
        lora_layers = LoRALayers.from_spec(flux=flux, training_spec=training_spec)

        # Bereid de fine-tuning dataset voor en maak de iterator
        dataset = Dataset.prepare_dataset(
            flux=flux,
            raw_data=training_spec.examples,
            width=training_spec.width,
            height=training_spec.height,
        )
        iterator = Iterator.from_spec(
            training_spec=training_spec,
            dataset=dataset
        )

        # Setup loss statistieken
        statistics = Statistics.from_spec(training_spec=training_spec)

        # De training state bestaat uit alles wat verandert tijdens training
        training_state = TrainingState(
            optimizer=optimizer,
            lora_layers=lora_layers,
            iterator=iterator,
            statistics=statistics,
        )

        return flux, runtime_config, training_spec, training_state


class CustomFlux1(Flux1):
    def __init__(
        self,
        model_config: ModelConfig,
        quantize: int | None = None,
        local_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        """
        Een aangepaste Flux1 klasse die gebruik maakt van onze eigen modellen.
        
        Args:
            model_config: De ModelConfig
            quantize: Kwantisatie bits (3, 4, 6, 8)
            local_path: Pad naar het lokale model
            lora_paths: Paden naar LoRA weights
            lora_scales: Schalen voor de LoRA weights
        """
        # Roep direct nn.Module.__init__ aan in plaats van Flux1.__init__
        from mlx import nn
        nn.Module.__init__(self)
        
        # Stel environment variables in om authenticatie over te slaan
        os.environ["HF_TOKEN"] = ""
        os.environ["HUGGING_FACE_HUB_TOKEN"] = ""
        
        # Importeer FluxInitializer met een lokaal pad
        try:
            from mflux.flux.flux_initializer import FluxInitializer
        except ModuleNotFoundError:
            from mflux.models.flux.flux_initializer import FluxInitializer
        
        # Initialiseer het model
        FluxInitializer.init(
            flux_model=self,
            model_config=model_config,
            quantize=quantize,
            local_path=local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )


def main():
    parser = argparse.ArgumentParser(description="Aangepaste Dreambooth finetuning met AITRADER modellen")
    
    # Algemene argumenten
    parser.add_argument("--train-config", type=str, required=False, help="Pad naar het training configuratie bestand")
    parser.add_argument("--train-checkpoint", type=str, required=False, help="Pad naar het checkpoint bestand om training te hervatten")
    parser.add_argument("--model", type=str, default="dev", choices=list(MODELS.keys()), help="Type model om te gebruiken")
    parser.add_argument("--path", type=str, default=None, help="Pad naar het lokale model")
    parser.add_argument("--quantize", type=int, choices=[3, 4, 6, 8], default=None, help="Kwantisatie bits")
    parser.add_argument("--low-ram", action="store_true", help="Gebruik minder RAM tijdens training")
    
    args = parser.parse_args()
    
    if args.low_ram:
        # Beperk MLX cache grootte
        os.environ["MLX_ALLOCATOR_CACHE_SIZE"] = "2147483648"  # 2GB
    
    # Initialiseer Dreambooth
    flux, runtime_config, training_spec, training_state = CustomDreamBoothInitializer.initialize(
        config_path=args.train_config,
        checkpoint_path=args.train_checkpoint,
        model_type=args.model,
        local_model_path=args.path,
        quantize=args.quantize
    )
    
    # Start training
    try:
        DreamBooth.train(
            flux=flux,
            runtime_config=runtime_config,
            training_spec=training_spec,
            training_state=training_state
        )
    except StopTrainingException as stop_exc:
        training_state.save(training_spec)
        print(stop_exc)


if __name__ == "__main__":
    main() 
