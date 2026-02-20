"""
Component Registry — enables loose coupling between PHILOS modules.

Each module (Perception, Cognitive, Learning, Control) registers its implementation
via the registry. Consumers look up components by interface type, not concrete class.
This allows swapping implementations (e.g., different RL algorithms, VLM backends)
without changing calling code.

Usage:
    # Register
    registry = ComponentRegistry()
    registry.register("perception", "yolo_world", YoloWorldDetector)

    # Resolve
    detector = registry.create("perception", "yolo_world", config=my_config)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ComponentRegistry:
    """Global registry for loosely-coupled PHILOS components.

    Categories:
        - perception: Detection and vision backends
        - cognitive: VLM, reward shaping, context encoding
        - learning: RL policies and algorithms
        - control: MPC solvers, safety shields
        - simulation: Isaac Sim environments
        - evaluation: Benchmark and metric implementations
    """

    _instance: ComponentRegistry | None = None
    _registry: dict[str, dict[str, type]]

    def __init__(self) -> None:
        self._registry = {}

    @classmethod
    def get_instance(cls) -> ComponentRegistry:
        """Singleton access to the global registry."""
        if cls._instance is None:
            cls._instance = ComponentRegistry()
        return cls._instance

    def register(self, category: str, name: str, cls_or_factory: type | Callable) -> None:
        """Register a component implementation.

        Args:
            category: Module category (e.g., 'perception', 'learning').
            name: Unique name within the category (e.g., 'yolo_world', 'ppo').
            cls_or_factory: The class or factory callable.
        """
        if category not in self._registry:
            self._registry[category] = {}
        self._registry[category][name] = cls_or_factory
        logger.info(f"Registered component: {category}/{name}")

    def create(self, category: str, name: str, **kwargs: Any) -> Any:
        """Create a component instance by category and name.

        Args:
            category: Module category.
            name: Component name.
            **kwargs: Constructor arguments.

        Returns:
            An instance of the requested component.

        Raises:
            KeyError: If the component is not registered.
        """
        if category not in self._registry or name not in self._registry[category]:
            available = self.list_components(category)
            raise KeyError(
                f"Component '{category}/{name}' not found. "
                f"Available in '{category}': {available}"
            )
        factory = self._registry[category][name]
        return factory(**kwargs)

    def list_components(self, category: str | None = None) -> dict[str, list[str]]:
        """List all registered components, optionally filtered by category."""
        if category:
            return {category: list(self._registry.get(category, {}).keys())}
        return {cat: list(impls.keys()) for cat, impls in self._registry.items()}

    def has(self, category: str, name: str) -> bool:
        """Check if a component is registered."""
        return category in self._registry and name in self._registry[category]


def register_component(category: str, name: str) -> Callable:
    """Decorator to auto-register a class with the global registry.

    Usage:
        @register_component("learning", "ppo")
        class PPOPolicy(BasePolicy):
            ...
    """

    def decorator(cls: type) -> type:
        ComponentRegistry.get_instance().register(category, name, cls)
        return cls

    return decorator
