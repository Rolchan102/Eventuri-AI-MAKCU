import time
import threading
from config import config


class JitterController:
    def __init__(self, makcu_instance):
        self.makcu = makcu_instance
        self.is_active = False
        self._thread = None  # ← храним ссылку на поток

        # Параметры (по умолчанию)
        self.jitter_interval = getattr(config, 'jitter_interval', 0.01)
        self.hipfire_sens = getattr(config, 'hipfire_sens', 1.0)
        self.ads_sens = getattr(config, 'ads_sens', 1.0)
        self.fine_tune_offset = getattr(config, 'fine_tune_offset', 0)

    def _calculate_base(self):
        """Расчёт базовой силы компенсации"""
        try:
            return int((6 / (self.hipfire_sens * self.ads_sens)) + 3 + self.fine_tune_offset)
        except (ZeroDivisionError, TypeError):
            return 5

    def _calculate_decline_threshold(self):
        """Порог начала затухания (в итерациях)"""
        try:
            return int(10 * self.hipfire_sens)
        except (TypeError, ValueError):
            return 10

    def _jitter_loop(self):
        """Синхронный цикл компенсации (запускается в отдельном потоке)"""
        base = self._calculate_base()
        decline = base * 12
        threshold = self._calculate_decline_threshold()
        counter = 0

        print(f"[JITTER] Start: base={base}, threshold={threshold}, interval={self.jitter_interval}")

        while self.is_active:
            if not self.makcu:
                time.sleep(0.1)
                continue

            try:
                # Zigzag-паттерн: синхронные вызовы (БЕЗ await!)
                self.makcu.move(base, -base)
                time.sleep(self.jitter_interval)
                self.makcu.move(-base, base)
                time.sleep(self.jitter_interval)

                counter += 2
                if counter >= threshold and decline > 0:
                    # Микро-коррекция + затухание
                    correction = max(1, int(decline / 18))
                    self.makcu.move(0, correction)
                    decline = int(decline * 0.85)
                    counter = 0

            except Exception as e:
                print(f"[JITTER] Error: {e}")
                time.sleep(0.1)

    def start(self):
        """Запустить компенсацию в отдельном потоке"""
        if self.is_active:
            return
        self.is_active = True
        # Запуск в отдельном потоке (daemon=True для авто-остановки при выходе)
        self._thread = threading.Thread(target=self._jitter_loop, daemon=True)
        self._thread.start()
        print("[JITTER] Thread started")

    def stop(self):
        """Остановить компенсацию"""
        self.is_active = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)  # Ждём до 0.5с завершения потока
        print("[JITTER] Stopped")

    def update_params(self):
        """Обновить параметры из config (вызывать при изменении настроек)"""
        self.jitter_interval = getattr(config, 'jitter_interval', 0.01)
        self.hipfire_sens = getattr(config, 'hipfire_sens', 1.0)
        self.ads_sens = getattr(config, 'ads_sens', 1.0)
        self.fine_tune_offset = getattr(config, 'fine_tune_offset', 0)
        print(f"[JITTER] Params updated: interval={self.jitter_interval}, hipfire={self.hipfire_sens}, ads={self.ads_sens}, fine_tune={self.fine_tune_offset}")
