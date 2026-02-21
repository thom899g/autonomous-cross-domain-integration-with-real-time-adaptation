import logging
from typing import Dict, Any
from adapters.messagingAdapter import MessagingAdapter
from data_normalizer.dataNormalizer import DataNormalizer
from ml_model_manager.mlModelManager import MLModelManager
from monitors.healthMonitor import HealthMonitor

class MasterAgent:
    """The central coordinating agent for the integration layer.
    
    Attributes:
        config: Configuration parameters for subsystems.
        subsystems: Dictionary mapping names to their respective components.
        logger: Logger instance for logging events and errors.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the MasterAgent with given configuration."""
        self.config = config
        self.subsystems: Dict[str, object] = {}
        self.logger = logging.getLogger("MasterAgent")
        
        # Initialize core components
        self._initialize_subsystems()
    
    def _initialize_subsystems(self) -> None:
        """Set up all subsystems necessary for operation."""
        try:
            # Data Handling
            self.subsystems['data_normalizer'] = DataNormalizer(
                config=self.config.get('data', {})
            )
            
            # Machine Learning Models
            self.subsystems['ml_model_manager'] = MLModelManager(
                config=self.config.get('models', {})
            )
            
            # Integration Adapters
            messaging_config = self.config.get('messaging', {})
            self.subsystems['messaging_adapter'] = MessagingAdapter(messaging_config)
            
            # Monitoring Services
            self.subsystems['health_monitor'] = HealthMonitor()
            
        except Exception as e:
            self._log_error(f"Failed to initialize subsystems: {str(e)}")
            raise InitializationError("Critical failure in initializing core components.")
    
    def _log_error(self, message: str) -> None:
        """Log an error with context."""
        self.logger.error(f"[ERROR] MasterAgent: {message}")
    
    def process_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming data through the integration layer.
        
        Args:
            raw_data: Raw input data to be processed.
            
        Returns:
            Processed and analyzed output data.
            
        Raises:
            ProcessingError: If any step in processing fails.
        """
        try:
            # Normalize data
            normalized_data = self.subsystems['data_normalizer'].normalize(raw_data)
            
            # Run predictive analytics
            predictions = self.subsystems['ml_model_manager'].predict(normalized_data)
            
            # Adapt based on feedback loops
            adapted_output = self.adapt(predictions, raw_data)
            
            return adapted_output
            
        except Exception as e:
            error_msg = f"Data processing failed: {str(e)}"
            self._log_error(error_msg)
            raise ProcessingError(error_msg) from e
    
    def adapt(self, predictions: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt system parameters based on current state and feedback.
        
        Args:
            predictions: Output from predictive models.
            input_data: Original input data for context.
            
        Returns:
            Adapted system configuration.
        """
        try:
            # Use health monitor to assess current state
            current_health = self.subsystems['health_monitor'].get_health()
            
            # Apply adaptive middleware logic
            adapted_params = self.subsystems['messaging_adapter'].adapt(
                predictions=predictions,
                health=current_health,
                input=input_data
            )
            
            return adapted_params
            
        except Exception as e:
            error_msg = f"Adaptation failed: {str(e)}"
            self._log_error(error_msg)
            raise AdaptationError(error_msg) from e
    
    def monitor(self) -> Dict[str, Any]:
        """Monitor the health and performance of connected systems.
        
        Returns:
            Dictionary with health metrics.
        """
        try:
            return self.subsystems['health_monitor'].monitor()
            
        except Exception as e:
            error_msg = f"Monitoring failed: {str(e)}"
            self._log_error(error_msg)
            raise MonitoringError(error_msg) from e
    
    def update_model(self, new_model_path: str) -> None:
        """Update the predictive model with a new version.
        
        Args:
            new_model_path: Path to the new model file.
        """
        try:
            self.subsystems['ml_model_manager'].update_model(new_model_path)
            
        except Exception as e:
            error_msg = f"Failed to update model from {new_model_path}"
            self._log_error(error_msg)
            raise ModelUpdateError(error_msg) from e
    
    def handle_error(self, error: Exception) -> None:
        """Centralized error handling and recovery."""
        try:
            # Log the error
            self._log_error(str(error))
            
            # Attempt recovery based on error type
            if isinstance(error, DataProcessingError):
                self.subsystems['data_normalizer'].recover()
            elif isinstance(error, ModelPredictionError):
                self.update_model(self.config['models']['fallback_model'])
            
        except Exception as e:
            critical_error_msg = f"Failed to handle error: {str(e)}"
            self.logger.critical(critical_error_msg)
            raise CriticalError("Unhandled system failure.")