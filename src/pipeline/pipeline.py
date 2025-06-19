class PipelineStep:
    def process(self, data):
        raise NotImplementedError

class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def run(self, data):
        for step in self.steps:
            data = step.process(data)
        return data
