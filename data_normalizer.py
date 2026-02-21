import logging
from typing import Dict, Any
import pandas as pd

class DataNormalizer:
    """Handles normalization of raw data inputs.
    
    Attributes:
        config: Configuration parameters for normalization.
        logger: Logger instance for logging events and errors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the DataNormalizer with given configuration."""
        self.config = config
        self.logger = logging.getLogger("DataNormalizer")
        
    def normalize(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Normalize and preprocess raw data.
        
        Args:
            raw_data: Dictionary of raw input data.
            
        Returns:
            DataFrame with normalized data.
            
        Raises:
            NormalizationError: If normalization fails.
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(raw_data)
            
            # Apply transformations based on config
            if self.config.get('normalize'):
                df = df.apply(self._normalize_column, axis=0)
                
            return df
            
        except Exception as e:
            error_msg = f"Normalization failed: {str(e)}"
            self.logger.error(error_msg)
            raise NormalizationError(error_msg) from e
    
    def _normalize_column(self, series: pd.Series) -> pd.Series:
        """Normalize a single column's data.
        
        Args:
            series: Input