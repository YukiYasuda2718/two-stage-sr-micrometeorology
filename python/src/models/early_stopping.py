import numpy as np
from logging import getLogger

logger = getLogger()


class EarlyStopping:
    def __init__(self, early_stopping_patience: int, **kwargs):
        assert early_stopping_patience > 0
        self.count = 0
        self.best_loss = np.inf
        self.patience = early_stopping_patience
        logger.info(f"Early stopping patience = {self.patience}")

    def __call__(self, current_loss: float) -> bool:
        if current_loss > self.best_loss:
            self.count += 1

            if self.count >= self.patience:
                logger.info(f"Early stopped: count = {self.count} (>= {self.patience})")
                return True

            logger.info(f"Early stopping count = {self.count}")

        else:
            self.count = 0
            self.best_loss = current_loss
            logger.info("Early stopping count is reset.")

        return False
