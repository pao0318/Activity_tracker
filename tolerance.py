# Getting tolerance level
class Tolerance:
    def __init__(self, str):
        self.str = str

    def get_tolerance(self):
        if str == "mild":
            tol_angle = 5
        else:
            tol_angle = 15
        return tol_angle


def get_tolerance(str):
    if str == "mild":
        tol_angle = 5
    else:
        tol_angle = 15
    return tol_angle
