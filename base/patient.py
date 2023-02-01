from event import Event
from info import Info
import datetime
        
        
class Patient():
    def __init__(self, id: str, events: list[Event]=None, info: Info=None) -> None:
        self.id = id
        self.events = events if events is not None else []

        if info is None:
            info = Info(id)
        self.set_info(info)

    def sort_events(self):
        self.events = sorted(self.events, key=lambda event: 
            datetime.strptime(event.get_data()['timestamp'][:10], "%d-%m-%Y")
        )

    def add_event(self, event):
        if self.id != event.get_id():
            raise ValueError('Patient id does not match event id')
        self.events.append(event)

    def set_info(self, info: Info):
        if self.id != info.get_id():
            raise ValueError('Patient id does not match info id')
        self.info = info

    def set_events(self, events: list[Event]):
        for event in events:
            self.add_event(event)

    def get_id(self):
        return self.id

    def get_events(self):
        return self.events

    def get_info(self):
        return self.info

    def __repr__(self) -> str:
        return f'Patient(id={self.id}, events={self.events}, info={self.info})'

