class SuccesLine:
    def __init__(self, i, line, start, end):
        self.line = line
        self.i = i
        self.start = start
        self.end = end

    def check_collision(self, rect):
        return bool(self.line.colliderect(rect))
