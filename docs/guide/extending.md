# Extending Inferno

## Updaters
All updaters should be designed to take in Layer objects, or sequences thereof, as the basic trainable unit. The updaters included with Inferno utilize the monitors in `inferno.observe` to maintain records of state. The behavior of `VirtualLayer` objects produced by `CompositeLayer` means that monitors should not be eagerly cleared. Only when the state of monitors across all layers are no longer needed is it safe to clear the state of any monitors.