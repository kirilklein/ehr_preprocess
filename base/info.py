from collections import UserDict


class Info(UserDict):
    def __init__(self, id: str, initialdata: dict=None) -> None:
        self.id = id
        super().__init__(initialdata)

    def __repr__(self) -> str:
        repr_str = f'PatientInfo(id={self.id}, data={super().__repr__()})'
        return repr_str

    def get_id(self):
        return self.id

    def get_data(self):
        return self.data.copy()

