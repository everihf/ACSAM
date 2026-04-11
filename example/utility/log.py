import logging
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


class BeijingTimeFormatter(logging.Formatter):
    """Formatter that renders timestamps in Asia/Shanghai time."""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=ZoneInfo("Asia/Shanghai"))
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def build_logger(name: str, log_path: Path) -> logging.Logger:
    """Create an isolated logger that writes to its own file and stdout."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = BeijingTimeFormatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


class Log:
    def __init__(self, log_each: int, initial_epoch: int = -1, logger=None):
        self.best_accuracy = 0.0
        self.log_each = int(log_each)
        self.epoch = int(initial_epoch)
        self.logger = logger or logging.getLogger(__name__)
        self.current_train_summary = None
        self.step = 0
        self.is_train = True
        self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}
        self.epoch_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}
        self.learning_rate = 0.0
        self.start_time = time.time()
        self.len_dataset = 0

    def train(self, len_dataset: int, reset_step: bool = True, reset_last_steps: bool = True) -> None:
        self.epoch += 1
        if self.epoch == 0:
            self._print_header()
        else:
            self.flush()

        self.is_train = True
        if reset_last_steps:
            self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}
        self._reset(len_dataset, reset_step=reset_step)

    def eval(self, len_dataset: int, reset_step: bool = True) -> None:
        self.flush()
        self.is_train = False
        self._reset(len_dataset, reset_step=reset_step)

    def __call__(self, model, loss, accuracy, learning_rate: float = None) -> None:
        if self.is_train:
            self._train_step(model, loss, accuracy, learning_rate)
        else:
            self._eval_step(loss, accuracy)

    def flush(self) -> None:
        if self.epoch_state["steps"] == 0:
            return

        if self.is_train:
            loss = self.epoch_state["loss"] / self.epoch_state["steps"]
            accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]
            self.current_train_summary = {
                "epoch": self.epoch,
                "loss": loss,
                "accuracy": accuracy,
                "learning_rate": self.learning_rate,
                "elapsed": self._time(),
            }
            return

        loss = self.epoch_state["loss"] / self.epoch_state["steps"]
        accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]

        if self.current_train_summary is not None:
            self.logger.info(
                f"|{self.current_train_summary['epoch']:12d}  "
                f"|{self.current_train_summary['loss']:12.4f}  "
                f"|{100 * self.current_train_summary['accuracy']:10.2f} %  "
                f"|{self.current_train_summary['learning_rate']:12.3e}  "
                f"|{self.current_train_summary['elapsed']:>12}  "
                f"|{loss:12.4f}  "
                f"|{100 * accuracy:10.2f} %  |"
            )

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy

    def _train_step(self, model, loss, accuracy, learning_rate: float) -> None:
        self.learning_rate = learning_rate if learning_rate is not None else self.learning_rate
        batch_steps = int(accuracy.numel())

        self.last_steps_state["loss"] += loss.sum().item()
        self.last_steps_state["accuracy"] += accuracy.sum().item()
        self.last_steps_state["steps"] += batch_steps

        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += batch_steps

        self.step += 1
        if self.log_each <= 0:
            return
        if self.step % self.log_each != self.log_each - 1:
            return

        loss_avg = self.last_steps_state["loss"] / self.last_steps_state["steps"]
        acc_avg = self.last_steps_state["accuracy"] / self.last_steps_state["steps"]
        self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}

        self.logger.info(
            f"|{self.epoch:12d}  "
            f"|{loss_avg:12.4f}  "
            f"|{100 * acc_avg:10.2f} %  "
            f"|{self.learning_rate:12.3e}  "
            f"|{self._time():>12}  |"
        )

    def _eval_step(self, loss, accuracy) -> None:
        batch_steps = int(accuracy.numel())
        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += batch_steps

    def _reset(self, len_dataset: int, reset_step: bool = True) -> None:
        self.start_time = time.time()
        if reset_step:
            self.step = 0
        self.len_dataset = len_dataset
        self.epoch_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}

    def _time(self) -> str:
        elapsed_seconds = int(time.time() - self.start_time)
        return f"{elapsed_seconds // 60:02d}:{elapsed_seconds % 60:02d} min"

    def _print_header(self) -> None:
        self.logger.info(
            "|------------|------------|------------|------------|------------|------------|------------|"
        )
        self.logger.info(
            "|   epoch    |    loss    |  accuracy  |    l.r.    |  elapsed   |    loss    |  accuracy  |"
        )
        self.logger.info(
            "|------------|------------|------------|------------|------------|------------|------------|"
        )
