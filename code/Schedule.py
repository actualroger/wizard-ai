
import numpy as np

# define Schedule class from which all other schedules inherit
class Schedule(object):
    def __init__(self, start_value, end_value, duration):
        # transitions from start @ 0 to end @ duration
        self.start_value = start_value
        self.end_value = end_value
        self.duration = duration

    def getValue(self, t) -> float:
        raise NotImplementedError("Schedule subclass must implement getValue(t)->float")

# create arbitrary schedule subclass from params
def createSchedule(params) -> Schedule:
    match params['schedule_type']:
        case 'linear':
            schedule = LinearSchedule(
                start_value=params['start_value'],
                end_value=params['end_value'],
                duration=params['duration'])
        case 'exponential':
            schedule = ExponentialSchedule(
                start_value=params['start_value'],
                end_value=params['end_value'],
                duration=params['duration'])
        case _:
            schedule = None
    return schedule

class LinearSchedule(Schedule):
    def __init__(self, start_value, end_value, duration):
        super().__init__(start_value, end_value, duration)
        # difference between the start value and the end value
        self.schedule_amount = end_value - start_value

    def getValue(self, time) -> float:
        # logic: if time > duration, use the end value, else use the scheduled value
        return self.start_value + self.schedule_amount * min(1.0, time * 1.0 / self.duration)

class ExponentialSchedule(Schedule):
    def __init__(self, start_value, end_value, duration):
        super().__init__(start_value, end_value, duration)
        # value = a * exp(b * t)
        self.a = self.start_value
        self.b = - np.log(self.start_value / self.end_value) / (self.duration - 1)

    def getValue(self, step) -> float:
        # start_value, if step == 0 or less
        # end_value, if step == duration - 1 or more
        return self.start_value if step <= 0 else self.end_value if step >= self.duration - 1 else self.a * np.exp(self.b * step)
