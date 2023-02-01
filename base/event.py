from collections import UserDict

class Event(UserDict):
    def __init__(self, 
        id: str,
        concept: str,
        timestamp: str,
        admission_id: str=None, 
        value: float=None, 
        unit: str=None
    ) -> None:
        self.id = id

        super().__init__({
            'concept': concept,
            'timestamp': timestamp,
            'admission_id': admission_id,
            'value': value,
            'unit': unit
        })

    def get_id(self):
        return self.id

    def get_data(self):
        return self.data.copy()

    def __repr__(self) -> str:
        return f'Event(id={self.id}, data={super().__repr__()})'

