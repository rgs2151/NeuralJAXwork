"""Count-weighted epoch metrics."""


class ClassificationAccumulator:
    def __init__(self):
        self.loss_sum = 0.0
        self.correct = 0
        self.count = 0

    def update(self, result):
        count = int(result["count"])
        if count:
            self.loss_sum += float(result["loss"]) * count
            self.correct += int(result.get("correct", 0))
            self.count += count

    def compute(self):
        if not self.count:
            raise ValueError("No valid examples were accumulated")
        return {"loss": self.loss_sum / self.count, "accuracy": self.correct / self.count}
