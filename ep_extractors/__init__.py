from typing import Dict, Type, List
from ep_extractors.base import BaseEpistemicExtractor

class UQExtractorRegistry:
    """
    Extensible factory registry mapping UQ approach names to their 
    respective BaseEpistemicExtractor implementation classes.
    """
    _registry: Dict[str, Type[BaseEpistemicExtractor]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a new BaseEpistemicExtractor class.
        
        Parameters:
        -----------
        name : str
            Unique registration key for the approach.
        """
        def decorator(subclass: Type[BaseEpistemicExtractor]):
            if not issubclass(subclass, BaseEpistemicExtractor):
                raise TypeError(
                    f"Class {subclass.__name__} must subclass BaseEpistemicExtractor to be registered."
                )
            if name in cls._registry:
                raise ValueError(
                    f"UQ Extractor key '{name}' is already registered by class {cls._registry[name].__name__}."
                )
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str, model, **kwargs) -> BaseEpistemicExtractor:
        """
        Retrieves and instantiates a registered UQ extractor by its name.
        
        Parameters:
        -----------
        name : str
            Registration key of the desired approach.
        model : RandomForestRegressor
            The fitted Random Forest surrogate model.
        **kwargs : dict
            Arguments passed to the constructor of the class.
            
        Returns:
        --------
        extractor : BaseEpistemicExtractor
            An instance of the registered UQ extractor.
        """
        if name not in cls._registry:
            raise KeyError(
                f"No UQ Extractor is registered under the key '{name}'. "
                f"Available extractors: {cls.list_registered()}"
            )
        return cls._registry[name](model, **kwargs)

    @classmethod
    def list_registered(cls) -> List[str]:
        """
        Lists all registered UQ extractor keys.
        
        Returns:
        --------
        keys : list of str
            List of registered approach names.
        """
        return list(cls._registry.keys())

# Import all extractor modules to trigger their @register decorators
from ep_extractors import standard_disagreement
from ep_extractors import chen_variance
from ep_extractors import shaker_entropy
from ep_extractors import likelihood_credal
from ep_extractors import standard_proximity
from ep_extractors import proximity_b
from ep_extractors import proximity_bc
from ep_extractors import proximity_auto_lambda

