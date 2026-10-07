import csv
from datetime import datetime
import os
from pathlib import Path

class DataLogger:
    def __init__(self, log_dir: str | Path | None = None):
        self.log_dir = Path(log_dir or os.environ.get("VQVLA_LOG_DIR", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"ratio_{self.timestamp}.csv"

        # 定义CSV表头
        self.header = [
            "total_step", "exe_step", "tran_step", "exe_rate", "tran_rate"
        ]

        # 初始化文件（仅在第一次运行时写入表头）
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def log_step(self, total_step, exe_step, tran_step):
        """记录单步推理数据"""
        record = [total_step, exe_step, tran_step, exe_step / total_step, tran_step / total_step]

        # 安全写入CSV（使用追加模式）
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(record)
        except (PermissionError, IOError) as e:
            print(f"写入日志失败: {str(e)}")
