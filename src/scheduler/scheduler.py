# Import the discrete event scheduler as the main scheduler
from .discrete_event_scheduler import DiscreteEventScheduler as Scheduler
from .discrete_event_scheduler import SchedulingEvent, SchedulingEventType

# For backward compatibility, expose the discrete event scheduler
__all__ = ['Scheduler', 'SchedulingEvent', 'SchedulingEventType']
