import time
import logging


class Log:
    def __init__(self, log_each: int, initial_epoch=-1, logger=None):
        self.best_accuracy = 0.0  # 最好的验证集准确率。
        self.log_each = log_each
        self.epoch = initial_epoch
        self.logger = logger or logging.getLogger(__name__)
        self.current_train_summary = None

    def train(self, len_dataset: int) -> None:
        self.epoch += 1
        if self.epoch == 0:
            self._print_header()  # 第一轮就打印表头
        else:
            self.flush()  # 否则，先把上一轮训练/验证结果完整打印出来

        self.is_train = True
        self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}
        self._reset(len_dataset)  # 开始新的训练阶段前，把时间、step 计数、epoch 累计统计都清零。

    def eval(self, len_dataset: int) -> None:
        self.flush()
        self.is_train = False
        self._reset(len_dataset)

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
        else:
            loss = self.epoch_state["loss"] / self.epoch_state["steps"]
            accuracy = self.epoch_state["accuracy"] / self.epoch_state["steps"]

            if self.current_train_summary is not None:
                self.logger.info(
                    f"┃{self.current_train_summary['epoch']:12d}  ┃"
                    f"{self.current_train_summary['loss']:12.4f}  │"
                    f"{100 * self.current_train_summary['accuracy']:10.2f} %  ┃"
                    f"{self.current_train_summary['learning_rate']:12.3e}  │"
                    f"{self.current_train_summary['elapsed']:>12}  ┃"
                    f"{loss:12.4f}  │{100 * accuracy:10.2f} %  ┃"
                )

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy

    def _train_step(self, model, loss, accuracy, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.last_steps_state["loss"] += loss.sum().item()
        self.last_steps_state["accuracy"] += accuracy.sum().item()
        self.last_steps_state["steps"] += loss.size(0)
        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += loss.size(0)
        self.step += 1

        if self.step % self.log_each == self.log_each - 1:
            loss = self.last_steps_state["loss"] / self.last_steps_state["steps"]
            accuracy = self.last_steps_state["accuracy"] / self.last_steps_state["steps"]

            self.last_steps_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}

            self.logger.info(
                f"┃{self.epoch:12d}  ┃{loss:12.4f}  │{100 * accuracy:10.2f} %  ┃"
                f"{learning_rate:12.3e}  │{self._time():>12}  ┃"
            )

    def _eval_step(self, loss, accuracy) -> None:
        self.epoch_state["loss"] += loss.sum().item()
        self.epoch_state["accuracy"] += accuracy.sum().item()
        self.epoch_state["steps"] += loss.size(0)

    def _reset(self, len_dataset: int) -> None:
        self.start_time = time.time()
        self.step = 0
        self.len_dataset = len_dataset
        self.epoch_state = {"loss": 0.0, "accuracy": 0.0, "steps": 0}

    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def _print_header(self) -> None:
        self.logger.info("┏━━━━━━━━━━━━━━┳━━━━━━━╸T╺╸R╺╸A╺╸I╺╸N╺━━━━━━━┳━━━━━━━╸S╺╸T╺╸A╺╸T╺╸S╺━━━━━━━┳━━━━━━━╸V╺╸A╺╸L╺╸I╺╸D╺━━━━━━━┓")
        self.logger.info("┃              ┃              ╷              ┃              ╷              ┃              ╷              ┃")
        self.logger.info("┃       epoch  ┃        loss  │    accuracy  ┃        l.r.  │     elapsed  ┃        loss  │    accuracy  ┃")
        self.logger.info("┠──────────────╂──────────────┼──────────────╂──────────────┼──────────────╂──────────────┼──────────────┨")
