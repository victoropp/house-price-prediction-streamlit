# Code Architecture and Software Design

## Architecture Overview

The preprocessing pipeline follows a **modular, object-oriented architecture** designed for scalability, maintainability, and production deployment. Each component is designed with clear separation of concerns, comprehensive error handling, and extensive validation.

## 🏗️ System Architecture

### High-Level Design Pattern
```
┌─────────────────────────────────────────────────────┐
│                PREPROCESSING PIPELINE               │
├─────────────────────────────────────────────────────┤
│  Phase 1  │  Phase 2  │  Phase 3  │  Phase 4  │ ...│
│  Target   │  Missing  │ Feature   │   Dist.   │    │
│  Transform│  Values   │ Engineer  │Transform  │    │
├─────────────────────────────────────────────────────┤
│               SHARED COMPONENTS                     │
│  • Configuration Management                         │
│  • Validation Framework                             │  
│  • Visualization Engine                             │
│  • Error Handling System                           │
└─────────────────────────────────────────────────────┘
```

### Core Design Principles

1. **Single Responsibility**: Each phase handles one transformation type
2. **Open/Closed**: Easy to extend without modifying existing code
3. **Dependency Injection**: Configurable parameters through JSON
4. **Interface Segregation**: Clear contracts between phases
5. **Liskov Substitution**: Phases can be replaced/upgraded independently

## 📦 Component Structure

### Phase Template Pattern
Each preprocessing phase follows a standardized structure:

```python
class PhaseNPipeline:
    """Template for all preprocessing phases."""
    
    def __init__(self):
        """Initialize with default configuration."""
        self.config = {}
        self.validation_results = {}
        self.artifacts = {}
    
    def load_data(self) -> 'PhaseNPipeline':
        """Load input data from previous phase."""
        pass
    
    def analyze_data(self) -> 'PhaseNPipeline':
        """Analyze data characteristics for processing."""
        pass
    
    def apply_transformations(self) -> 'PhaseNPipeline':
        """Apply core transformations."""
        pass
    
    def validate_results(self) -> 'PhaseNPipeline':
        """Validate transformation results."""
        pass
    
    def generate_visualizations(self) -> 'PhaseNPipeline':
        """Create result visualizations.""" 
        pass
    
    def save_artifacts(self) -> 'PhaseNPipeline':
        """Save transformed data and configuration."""
        pass
    
    def run_complete_pipeline(self) -> 'PhaseNPipeline':
        """Execute full phase pipeline."""
        return (self
                .load_data()
                .analyze_data() 
                .apply_transformations()
                .validate_results()
                .generate_visualizations()
                .save_artifacts())
```

## 🔧 Technical Implementation

### Class Hierarchy and Inheritance
```python
# Base Pipeline Interface
class BasePreprocessingPipeline(ABC):
    """Abstract base class for all preprocessing phases."""
    
    @abstractmethod
    def load_data(self) -> 'BasePreprocessingPipeline':
        """Load and validate input data."""
        pass
    
    @abstractmethod 
    def process_data(self) -> 'BasePreprocessingPipeline':
        """Apply transformations to data."""
        pass
    
    @abstractmethod
    def validate_output(self) -> 'BasePreprocessingPipeline':
        """Validate processed data quality."""
        pass

# Concrete Implementation Example
class TargetTransformationPipeline(BasePreprocessingPipeline):
    """Phase 1: Target variable transformation implementation."""
    
    def __init__(self):
        super().__init__()
        self.transformation_methods = {
            'log1p': np.log1p,
            'sqrt': np.sqrt,
            'boxcox': self._apply_boxcox
        }
```

### Configuration Management System
```python
class ConfigurationManager:
    """Centralized configuration management."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return self._get_default_config()
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration with type conversion."""
        serializable_config = self._convert_to_serializable(config)
        with open(self.config_path, 'w') as f:
            json.dump(serializable_config, f, indent=4)
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy/pandas types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {self._convert_to_serializable(k): 
                    self._convert_to_serializable(v) 
                    for k, v in obj.items()}
        return obj
```

### Validation Framework
```python
class ValidationFramework:
    """Comprehensive data validation system."""
    
    def __init__(self):
        self.validators = {
            'data_quality': DataQualityValidator(),
            'statistical': StatisticalValidator(),
            'domain': DomainValidator(),
            'performance': PerformanceValidator()
        }
    
    def validate_phase_output(self, 
                             phase_name: str,
                             data: pd.DataFrame,
                             target: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Run all validators on phase output."""
        results = {'phase': phase_name, 'timestamp': datetime.now()}
        
        for validator_name, validator in self.validators.items():
            try:
                validator_result = validator.validate(data, target)
                results[validator_name] = validator_result
            except Exception as e:
                results[validator_name] = {'error': str(e), 'status': 'failed'}
        
        return results

class DataQualityValidator:
    """Validates basic data quality metrics."""
    
    def validate(self, data: pd.DataFrame, target: pd.Series = None) -> Dict[str, Any]:
        return {
            'missing_values': data.isnull().sum().sum(),
            'duplicate_rows': data.duplicated().sum(),
            'data_types': data.dtypes.value_counts().to_dict(),
            'shape': data.shape,
            'memory_usage': data.memory_usage(deep=True).sum(),
            'status': 'passed' if data.isnull().sum().sum() == 0 else 'warning'
        }
```

### Error Handling and Logging
```python
import logging
from functools import wraps

class PipelineLogger:
    """Centralized logging system for pipeline operations."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_phase_start(self, phase_name: str) -> None:
        """Log phase execution start."""
        self.logger.info(f"Starting {phase_name}")
    
    def log_phase_end(self, phase_name: str, success: bool = True) -> None:
        """Log phase execution completion."""
        status = "completed successfully" if success else "failed"
        self.logger.info(f"Phase {phase_name} {status}")

def handle_pipeline_errors(func):
    """Decorator for comprehensive error handling."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {func.__name__}: {str(e)}")
            # Store error for later analysis
            if not hasattr(self, 'errors'):
                self.errors = []
            self.errors.append({
                'function': func.__name__,
                'error': str(e),
                'timestamp': datetime.now()
            })
            raise
    return wrapper
```

### Visualization Engine
```python
class VisualizationEngine:
    """Centralized visualization generation system."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        plt.style.use(style)
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
        self.figure_size = (24, 18)
    
    def create_phase_dashboard(self, 
                              phase_name: str,
                              data: Dict[str, Any]) -> plt.Figure:
        """Create standardized dashboard for phase results."""
        fig = plt.figure(figsize=self.figure_size)
        
        # Title and metadata
        fig.suptitle(f'{phase_name} Results Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Grid layout based on data content
        self._create_grid_layout(fig, data)
        
        return fig
    
    def save_visualization(self, fig: plt.Figure, 
                          filename: str, dpi: int = 300) -> None:
        """Save visualization with consistent settings."""
        fig.tight_layout(pad=3.0)
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)  # Free memory
```

## 🗄️ Data Flow Architecture

### Inter-Phase Data Contracts
```python
class DataContract:
    """Defines data contracts between phases."""
    
    @dataclass
    class PhaseInput:
        """Input specification for each phase."""
        required_columns: List[str]
        optional_columns: List[str] = None
        data_types: Dict[str, str] = None
        constraints: Dict[str, Any] = None
    
    @dataclass
    class PhaseOutput:
        """Output specification for each phase.""" 
        guaranteed_columns: List[str]
        added_columns: List[str] = None
        removed_columns: List[str] = None
        transformations_applied: List[str] = None

# Example contract definitions
PHASE_CONTRACTS = {
    'phase1': DataContract.PhaseInput(
        required_columns=['SalePrice'],
        constraints={'SalePrice': {'min': 0, 'type': 'numeric'}}
    ),
    'phase2': DataContract.PhaseInput(
        required_columns=['SalePrice_transformed'],
        constraints={'missing_threshold': 0.5}
    )
}
```

### Pipeline State Management
```python
class PipelineState:
    """Manages pipeline execution state and recovery."""
    
    def __init__(self, state_file: str = 'pipeline_state.json'):
        self.state_file = Path(state_file)
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load pipeline state from file."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {'completed_phases': [], 'last_checkpoint': None}
    
    def mark_phase_complete(self, phase_name: str) -> None:
        """Mark phase as completed."""
        if phase_name not in self.state['completed_phases']:
            self.state['completed_phases'].append(phase_name)
            self.state['last_checkpoint'] = datetime.now().isoformat()
            self._save_state()
    
    def can_skip_phase(self, phase_name: str) -> bool:
        """Check if phase can be skipped."""
        return phase_name in self.state['completed_phases']
```

## 🔧 Performance Optimization

### Memory Management
```python
class MemoryOptimizer:
    """Optimizes memory usage during processing."""
    
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
    
    def process_in_chunks(self, data: pd.DataFrame, 
                         func: callable) -> pd.DataFrame:
        """Process large datasets in memory-efficient chunks."""
        results = []
        
        for chunk_start in range(0, len(data), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(data))
            chunk = data.iloc[chunk_start:chunk_end]
            
            # Process chunk
            processed_chunk = func(chunk)
            results.append(processed_chunk)
            
            # Force garbage collection
            del chunk
            gc.collect()
        
        return pd.concat(results, ignore_index=True)
```

### Parallel Processing
```python
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor

class ParallelProcessor:
    """Handles parallel processing of independent operations."""
    
    def __init__(self, n_jobs: int = -1):
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    
    def parallel_transform(self, data: pd.DataFrame,
                          transform_func: callable,
                          columns: List[str]) -> pd.DataFrame:
        """Apply transformations in parallel across columns."""
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {}
            
            # Submit jobs
            for col in columns:
                future = executor.submit(transform_func, data[col])
                futures[col] = future
            
            # Collect results
            results = {}
            for col, future in futures.items():
                results[col] = future.result()
        
        # Combine results
        result_df = data.copy()
        for col, transformed_data in results.items():
            result_df[f"{col}_transformed"] = transformed_data
        
        return result_df
```

## 📊 Testing Framework

### Unit Testing Structure
```python
import unittest
from unittest.mock import Mock, patch

class TestTargetTransformationPipeline(unittest.TestCase):
    """Unit tests for target transformation phase."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = TargetTransformationPipeline()
        self.sample_data = pd.DataFrame({
            'SalePrice': [100000, 200000, 150000, 300000]
        })
    
    def test_boxcox_transformation(self):
        """Test BoxCox transformation functionality."""
        result = self.pipeline._apply_boxcox(self.sample_data['SalePrice'])
        
        # Assertions
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)  # transformed_data, lambda
        self.assertTrue(np.isfinite(result[0]).all())
    
    def test_pipeline_integration(self):
        """Test complete pipeline execution."""
        with patch.object(self.pipeline, 'save_artifacts') as mock_save:
            result = self.pipeline.run_complete_pipeline()
            mock_save.assert_called_once()
            self.assertIsInstance(result, TargetTransformationPipeline)

class TestIntegration(unittest.TestCase):
    """Integration tests across phases."""
    
    def test_phase_data_flow(self):
        """Test data flow between consecutive phases."""
        # Phase 1 output should be valid Phase 2 input
        phase1 = TargetTransformationPipeline()
        phase1_output = phase1.run_complete_pipeline()
        
        phase2 = MissingValueTreatmentPipeline()
        phase2.train_data = phase1_output.train_data
        
        # Should not raise exception
        phase2.validate_input_data()
```

## 🚀 Deployment Architecture

### Production Deployment Pattern
```python
class ProductionPipeline:
    """Production-ready pipeline wrapper."""
    
    def __init__(self, config_path: str):
        self.config = self._load_production_config(config_path)
        self.monitoring = PipelineMonitoring()
        self.phases = self._initialize_phases()
    
    def process_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Process data through complete pipeline."""
        
        # Monitoring and logging
        self.monitoring.start_processing()
        
        try:
            current_data = input_data.copy()
            
            for phase_name, phase in self.phases.items():
                # Pre-phase validation
                self._validate_phase_input(phase_name, current_data)
                
                # Execute phase
                current_data = phase.process(current_data)
                
                # Post-phase validation  
                self._validate_phase_output(phase_name, current_data)
                
                # Update monitoring
                self.monitoring.record_phase_completion(phase_name)
            
            return current_data
            
        except Exception as e:
            self.monitoring.record_error(str(e))
            raise
        
        finally:
            self.monitoring.end_processing()
```

## 📈 Scalability Considerations

### Horizontal Scaling Pattern
```python
class DistributedPipeline:
    """Distributed processing for large datasets."""
    
    def __init__(self, cluster_config: Dict[str, Any]):
        self.cluster = self._initialize_cluster(cluster_config)
        self.data_partitioner = DataPartitioner()
    
    def process_large_dataset(self, data_path: str) -> str:
        """Process dataset across multiple nodes."""
        
        # Partition data
        partitions = self.data_partitioner.partition_file(data_path)
        
        # Distribute processing
        futures = []
        for partition in partitions:
            future = self.cluster.submit(self._process_partition, partition)
            futures.append(future)
        
        # Collect and merge results
        results = [future.result() for future in futures]
        return self._merge_results(results)
```

This architecture provides a robust, scalable, and maintainable foundation for the advanced preprocessing pipeline, ensuring production readiness and ease of extension.