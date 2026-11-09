"""main v3.0 - конвейер захват -> детекция -> замок цели -> наведение (+джиттер).

Что изменено относительно main.py (v2.12):
  * логирование через QueueHandler/QueueListener: запись в файл и консоль больше не тормозит цикл кадров;
  * убран time.sleep(0.001) в конце цикла (цикл и так блокируется на frame_queue.get);
  * блок калибровки EGO выключен по умолчанию (config.ego_calib_log = True вернёт, уровень INFO);
  * время кадра = момент ЗАХВАТА (t_cap), а не момент извлечения из очереди: корректны скорость цели,
    lost_s и EgoTracker; в STATUS добавлен age = задержка захват -> обработка;
  * поток захвата: отсев дубликатов кадров (FrameSync NDI возвращает последний кадр сколько угодно раз;
    раньше поток крутился на полной скорости и отнимал GIL), пульс 10 мс, чтобы статичная сцена не вешала цикл
    (fps в STATUS теперь = число УНИКАЛЬНЫХ кадров + пульс, а не число повторов одного кадра);
  * JitterController создаётся ДО запуска потоков и подключается к EgoTracker (раньше трекер создавался
    с jitter_offset=None, и джиттер не учитывался: в CONFIG было jitter=off);
  * ego получает ОТПРАВЛЕННОЕ смещение (после мёртвой зоны/ограничений, для bezier/silent/smooth - сумма пути),
    а не «желаемое»; на кадре без наведения оно обнуляется (раньше повторялось);
  * геометрия кадра и DetectorConfig кэшируются (пересчёт только при смене размеров/режима);
    множитель чувствительности пересчитывается при смене config.in_game_sens;
  * исключение внутри обработки кадра больше не убивает поток детекции (лог не чаще раза в секунду);
  * чистое состояние при каждом start() (очередь, EMA, замок, мёртвая зона, ego).
Публичный API для GUI не менялся: start_aimbot/stop_aimbot/..., fps, _aimbot_running, makcu.
Нужен detection_v2 >= 2.12.1 (EgoTracker.set_jitter_offset); на более старом работает через запасной путь.
"""
from __future__ import annotations

import atexit
import logging
import math
import os
import queue
import threading
import time
import zlib
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

import cv2
import numpy as np

from capture import get_camera
from config import config
from detection import get_stats, load_model, perform_detection
from detection_v2 import (
    ButtonDebounce,
    DeadzoneFilter,
    DetectorConfig,
    EgoTracker,
    LockConfig,
    LockStatus,
    TargetLifecycle,
    TargetLock,
    detect,
    draw_debug_overlay,
    log_startup_config,
)
from jitter import JitterController
from mouse import Mouse, is_button_pressed
from windmouse_smooth import smooth_aimer

__version__ = "3.0.0"

# Повторять кадр, идентичный предыдущему, не чаще чем раз в столько секунд (пульс для статичной сцены:
# нажатия кнопки, джиттер и замок обновляются по кадрам). 0.01 = до 100 кадров/с на неподвижной картинке.
# НЕ ставьте много: при 0.05 статичный тест давал ровно 20 fps. Настраивается config.capture_dup_heartbeat_s.
DUP_HEARTBEAT_S = 0.01


# === Логирование ===
_log_listener: QueueListener | None = None


def _setup_logging() -> None:
    """Файл и консоль обслуживает отдельный поток; горячий цикл только кладёт запись в очередь."""
    global _log_listener
    level_name = os.environ.get("AIMBOT_LOG_LEVEL", getattr(config, "log_level", "WARNING")).upper()
    level = getattr(logging, level_name, logging.WARNING)
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s | %(message)s", datefmt="%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)

    handlers: list[logging.Handler] = []
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    handlers.append(console)
    try:
        os.makedirs("logs", exist_ok=True)
        fh = RotatingFileHandler("logs/aimbot.log", maxBytes=5 * 1024 * 1024, backupCount=3,
                                 encoding="utf-8")
        fh.setFormatter(fmt)
        handlers.append(fh)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Не удалось создать файл логов: {e}")

    log_queue: queue.SimpleQueue = queue.SimpleQueue()
    root.addHandler(QueueHandler(log_queue))
    _log_listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    _log_listener.start()
    atexit.register(_log_listener.stop)


_setup_logging()
logger = logging.getLogger("aimbot")

# === Глобальное состояние для GUI ===
fps = 0.0
_aimbot_running = False
makcu = None


class Aimbot:
    def __init__(self):
        self.stop_event = threading.Event()
        self._running = False
        self.fps = 0.0

        self.capture_thread = None
        self.smooth_thread = None
        self.aim_thread = None

        self.frame_queue = queue.Queue(maxsize=1)           # элементы: (image, t_capture)
        self.smooth_move_queue = queue.Queue(maxsize=100)   # smooth-траектории и шаги джиттера

        self.makcu = None
        self._owns_makcu = False
        self._makcu_lock = threading.Lock()                 # smooth_movement_loop + JitterController
        self.jitter_controller = None

        self.model = None
        self.class_names = {}

        # EMA состояние
        self.ema_target_x = None
        self.ema_target_y = None
        self.jump_threshold_px = 250.0

        # Диагностика
        self._had_target = False
        self._last_best_center = None
        self._frame_ms_ema = 0.0
        self._age_ms_ema = 0.0
        self._slow_frames = 0
        self._lock_waits = 0
        self._ndi_no_frame_streak = 0
        self._dup_skipped = 0
        self._last_summary_ts = time.perf_counter()
        self._last_target_ts = 0.0
        self._last_dist = 999.0
        self._jitter_deactivate_ts = 0.0
        self._last_frame_ts = 0.0
        self._frame_dt = 0.014
        self._err_ts = 0.0

        # Отправленное наведение (единицы мыши) с прошлого кадра - для ego_shift
        self._last_aim_dx = 0.0
        self._last_aim_dy = 0.0

        # Кэши
        self._sens_cached = None
        self._distance_multiplier = 1.0
        self._geom_key = None
        self._geom = None
        self._cfg_cache = None

        # Окно отладки OpenCV (инициализация вне цикла)
        self._debug_window_initialized = False
        self._debug_win_name = "AI Debug"

        # === detection_v2 ===
        self.deadzone = DeadzoneFilter(
            enter=getattr(config, "deadzone_enter", 0.8),
            exit_ratio=0.6,
            radial=True,
            rescale=True,
        )

        self.button_debounce = ButtonDebounce(
            release_frames=getattr(config, "button_release_frames", 1),
            always_on=getattr(config, "always_on_aim", False),
        )

        # Джиттер подключается позже (контроллер создаётся в start(), когда есть makcu)
        self.ego_tracker = EgoTracker(
            px_per_unit=getattr(config, "ego_px_per_unit", 0.7),
            jitter_offset=None,
            latency_s=getattr(config, "ego_latency_s", 0.03),
            aim_latency_s=getattr(config, "aim_latency_s", 0.03),
            aim_smear_s=getattr(config, "aim_smear_s", 0.04),
        )

        lock_cfg = LockConfig.recommended(
            lost_frames=getattr(config, "lock_lost_frames", 15),
            match_radius=getattr(config, "lock_match_radius", 0.1),
            acquire_frames=getattr(config, "lock_acquire_frames", 2),
            acquire_conf=getattr(config, "lock_acquire_conf", 0.35),
            coast_radius_max=getattr(config, "lock_coast_radius_max", 1.0),
            hard_lock=getattr(config, "lock_hard_lock", True),
            radius_heights=getattr(config, "lock_radius_heights", 2.0),
            size_ratio_max=getattr(config, "lock_size_ratio_max", 2.0),
            neighbor_guard=getattr(config, "lock_neighbor_guard", 0.5),
            relock_after_lost=getattr(config, "lock_relock_after_lost", False),
            size_free_heights=getattr(config, "lock_size_free_heights", 0.5),
        )
        self.target_lock = TargetLock(lock_cfg)

        self.lifecycle = TargetLifecycle(
            lock=self.target_lock,
            debounce=self.button_debounce,
            priority_only=True,
            short_hold_frames=getattr(config, "short_hold_frames", 3),
            logger=logger,
        )

        priority_labels = set()
        if config.custom_player_label:
            priority_labels.add(config.custom_player_label.lower())
        if config.custom_head_label:
            priority_labels.add(config.custom_head_label.lower())

        self.detector_config = DetectorConfig(
            conf_threshold=getattr(config, "conf", 0.20),
            poi_offset=getattr(config, "player_poi_offset", 0.2),
            use_tracking=getattr(config, "use_tracking", False),
            priority_labels=frozenset(priority_labels),
            aim_margin_px=getattr(config, "aim_margin_px", 4.0),
            ref_center=None,
        )

    # ------------------------------------------------------------------ утилиты
    def _now_ms(self):
        return time.perf_counter() * 1000.0

    def _sens_multiplier(self) -> float:
        """Множитель расстояния от чувствительности в игре; пересчёт только при её смене."""
        sens = getattr(config, "in_game_sens", 1.5)
        if sens != self._sens_cached:
            self._distance_multiplier = 1.07437623 * math.pow(sens, -0.9936827126)
            self._sens_cached = sens
        return self._distance_multiplier

    def _geometry(self):
        """(ref_cx, ref_cy, region_left, region_top, crosshair_x, crosshair_y), DetectorConfig.
        Пересчитывается только при смене режима захвата или размеров."""
        mode = config.capturer_mode.lower()
        key = (mode, config.region_size, config.screen_width, config.screen_height,
               config.ndi_width, config.ndi_height, config.main_pc_width, config.main_pc_height)
        if key != self._geom_key:
            if mode == "mss":
                ref_cx = ref_cy = config.region_size / 2.0
                region_left = (config.screen_width - config.region_size) // 2
                region_top = (config.screen_height - config.region_size) // 2
                crosshair_x, crosshair_y = config.screen_width // 2, config.screen_height // 2
            else:
                ref_cx, ref_cy = config.ndi_width / 2.0, config.ndi_height / 2.0
                region_left = (config.main_pc_width - config.ndi_width) // 2
                region_top = (config.main_pc_height - config.ndi_height) // 2
                crosshair_x, crosshair_y = config.main_pc_width // 2, config.main_pc_height // 2
            dc = self.detector_config
            self._cfg_cache = DetectorConfig(
                conf_threshold=dc.conf_threshold,
                poi_offset=dc.poi_offset,
                use_tracking=dc.use_tracking,
                priority_labels=dc.priority_labels,
                aim_margin_px=dc.aim_margin_px,
                ref_center=(ref_cx, ref_cy),
            )
            self._geom = (ref_cx, ref_cy, region_left, region_top, crosshair_x, crosshair_y)
            self._geom_key = key
        return self._geom, self._cfg_cache

    def _reset_idle_state(self):
        smooth_aimer.reset_fatigue()
        self.ema_target_x = None
        self.ema_target_y = None
        self._last_dist = 999.0
        self.deadzone.reset()

    def _reset_runtime_state(self):
        """Чистое состояние перед запуском потоков (повторный start() после stop())."""
        try:
            while True:
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            while True:
                self.smooth_move_queue.get_nowait()
        except queue.Empty:
            pass
        self.ema_target_x = self.ema_target_y = None
        self._had_target = False
        self._last_best_center = None
        self._last_frame_ts = 0.0
        self._last_target_ts = 0.0
        self._last_aim_dx = self._last_aim_dy = 0.0
        self._ndi_no_frame_streak = 0
        self._dup_skipped = 0
        for obj in (self.deadzone, self.ego_tracker, self.target_lock):
            fn = getattr(obj, "reset", None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass

    def _ensure_jitter(self):
        """Создать JitterController (нужен makcu) и подключить его смещение к EgoTracker."""
        if self.jitter_controller is None:
            self.jitter_controller = JitterController(
                self.makcu,
                makcu_lock=self._makcu_lock,
                move_queue=self.smooth_move_queue,
            )
        offset_at = self.jitter_controller.offset_at
        setter = getattr(self.ego_tracker, "set_jitter_offset", None)
        if callable(setter):
            setter(offset_at)
        else:  # detection_v2 < 2.12.1
            self.ego_tracker._jitter = offset_at
            self.ego_tracker._last = None

    # ------------------------------------------------------------------ makcu
    # Вызывается ТОЛЬКО из smooth_movement_loop - единственный владелец makcu
    def _locked_move(self, dx, dy):
        if self.makcu is None:
            return
        t0 = time.perf_counter()
        with self._makcu_lock:
            wait_ms = (time.perf_counter() - t0) * 1000.0
            if wait_ms > 2.0:
                self._lock_waits += 1
            self.makcu.move(dx, dy)

    def _locked_click(self):
        if self.makcu is None:
            return
        t0 = time.perf_counter()
        with self._makcu_lock:
            wait_ms = (time.perf_counter() - t0) * 1000.0
            if wait_ms > 2.0:
                self._lock_waits += 1
            self.makcu.click()

    def smooth_movement_loop(self):
        """Единственный поток, обращающийся к makcu. Агрегирует микро-смещения."""
        logger.info("Smooth movement thread started (Sole owner of makcu)")
        while not self.stop_event.is_set():
            try:
                # Блокирующее ожидание первого элемента (экономит CPU)
                dx, dy, delay = self.smooth_move_queue.get()

                if delay > 0:
                    if dx != 0 or dy != 0:
                        self._locked_move(dx, dy)
                    time.sleep(delay)
                    continue

                sum_dx, sum_dy = dx, dy
                while True:
                    try:
                        ndx, ndy, ndelay = self.smooth_move_queue.get_nowait()
                        if ndelay > 0:
                            if sum_dx != 0 or sum_dy != 0:
                                self._locked_move(sum_dx, sum_dy)
                            if ndx != 0 or ndy != 0:
                                self._locked_move(ndx, ndy)
                            time.sleep(ndelay)
                            break
                        else:
                            sum_dx += ndx
                            sum_dy += ndy
                    except queue.Empty:
                        if sum_dx != 0 or sum_dy != 0:
                            self._locked_move(sum_dx, sum_dy)
                        break
            except Exception as e:  # noqa: BLE001
                logger.error(f"Smooth movement failed: {e}")
                time.sleep(0.001)
        logger.info("Smooth movement thread stopped")

    # ------------------------------------------------------------------ захват
    def _put_frame(self, image, t_cap):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.frame_queue.put_nowait((image, t_cap))
        except queue.Full:
            pass

    def capture_loop(self):
        camera, _ = get_camera()
        last_selected = None
        last_sources_update = 0.0
        last_sig = None
        last_put_ts = 0.0
        dedup = getattr(config, "capture_dedup", True)
        heartbeat = float(getattr(config, "capture_dup_heartbeat_s", DUP_HEARTBEAT_S))

        while not self.stop_event.is_set():
            try:
                now = time.perf_counter()
                if now - last_sources_update > 3.0:
                    try:
                        config.ndi_sources = camera.list_sources()
                        last_sources_update = now
                    except Exception:  # noqa: BLE001
                        config.ndi_sources = []

                if config.capturer_mode.lower() == "ndi":
                    desired = config.ndi_selected_source
                    if isinstance(desired, str) and desired in config.ndi_sources:
                        if (desired != last_selected) or not camera.connected:
                            camera.select_source(desired)
                            last_selected = desired
                            logger.info("NDI source selected: %s", desired)

                image = camera.get_latest_frame()
                if image is None:
                    time.sleep(0.001)
                    self._ndi_no_frame_streak += 1
                    if self._ndi_no_frame_streak == 60:
                        logger.warning("NDI: нет кадров от '%s' (60 опросов)", config.ndi_selected_source)
                    continue

                t_cap = time.perf_counter()
                if self._ndi_no_frame_streak:
                    logger.info("NDI frames resumed after %d empty polls", self._ndi_no_frame_streak)
                self._ndi_no_frame_streak = 0

                if dedup and image.flags.c_contiguous:
                    sig = zlib.adler32(image)
                    if sig == last_sig and (t_cap - last_put_ts) < heartbeat:
                        self._dup_skipped += 1
                        time.sleep(0.0005)
                        continue
                    last_sig = sig
                last_put_ts = t_cap
                self._put_frame(image, t_cap)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Capture loop failed: {e}")
                time.sleep(1)
        try:
            camera.stop()
        except Exception:  # noqa: BLE001
            pass
        logger.info("Capture loop stopped")

    # ------------------------------------------------------------------ наведение
    def _enqueue_path(self, path):
        """Положить путь в очередь; вернуть реально поставленную сумму (единицы мыши)."""
        sx = sy = 0.0
        for move_dx, move_dy, delay in path:
            try:
                self.smooth_move_queue.put_nowait((move_dx, move_dy, delay))
                sx += move_dx
                sy += move_dy
            except queue.Full:
                break
        return sx, sy

    def _apply_movement_mode(self, dx, dy, frame_count, best_target=None):
        """Все режимы пушат задачи в smooth_move_queue (Zero-Lock).
        Запоминает ОТПРАВЛЕННОЕ смещение для ego_shift следующего кадра."""
        sent_dx = sent_dy = 0.0

        if config.mode == "normal":
            dt = self._frame_dt
            dist_px = math.hypot(dx, dy)

            if dist_px > 80:
                base_gain = 0.55
            elif dist_px > 40:
                base_gain = 0.45
            elif dist_px > 15:
                base_gain = 0.35
            else:
                base_gain = 0.25

            tau_x = 0.010 / max(config.normal_x_speed, 0.01)
            tau_y = 0.010 / max(config.normal_y_speed, 0.01)
            gain_x = min(1.0 - math.exp(-dt / tau_x), base_gain)
            gain_y = min(1.0 - math.exp(-dt / tau_y), base_gain)

            dx *= gain_x
            dy *= gain_y

            max_step = max(6.0, 2600.0 * dt)
            step = math.hypot(dx, dy)
            if step > max_step:
                k = max_step / step
                dx *= k
                dy *= k

            final_dx, final_dy = self.deadzone.apply(dx, dy)
            if final_dx != 0.0 or final_dy != 0.0:
                try:
                    self.smooth_move_queue.put_nowait((final_dx, final_dy, 0.0))
                    sent_dx, sent_dy = final_dx, final_dy
                except queue.Full:
                    pass

        elif config.mode == "bezier":
            path_fn = getattr(smooth_aimer, "calculate_bezier_path", None)
            path = [(dx, dy, 0.0)] if path_fn is None else path_fn(dx, dy, config)
            sent_dx, sent_dy = self._enqueue_path(path)

        elif config.mode == "silent":
            path_fn = getattr(smooth_aimer, "calculate_silent_path", None)
            path = [(dx, dy, 0.0)] if path_fn is None else path_fn(dx, dy, config)
            sent_dx, sent_dy = self._enqueue_path(path)

        elif config.mode == "smooth":
            path = smooth_aimer.calculate_smooth_path(dx, dy, config)
            sent_dx, sent_dy = self._enqueue_path(path)

        self._last_aim_dx = sent_dx
        self._last_aim_dy = sent_dy

    def _process_jitter(self, button_held: bool):
        if not getattr(config, "jitter_enabled", False) or not self.jitter_controller:
            return

        trigger_btn = int(getattr(config, "jitter_trigger_button", 0))
        require_ads = getattr(config, "jitter_require_ads", False)
        is_trigger_pressed = is_button_pressed(trigger_btn)
        is_rmb_pressed = is_button_pressed(1) if require_ads else False

        is_activated = button_held and is_trigger_pressed and (not require_ads or is_rmb_pressed)
        now = time.perf_counter()

        if is_activated:
            self._jitter_deactivate_ts = 0.0
            if not self.jitter_controller.is_active:
                self.jitter_controller.update_params()
                if self.jitter_controller.start():
                    logger.info("JITTER START | pattern=%s amp=%d/%d",
                                self.jitter_controller.pattern,
                                self.jitter_controller.amp_x,
                                self.jitter_controller.amp_y)
                # иначе поток ещё останавливается - повторим на следующем кадре
        else:
            if self._jitter_deactivate_ts == 0.0:
                self._jitter_deactivate_ts = now

            if self.jitter_controller.is_active and (now - self._jitter_deactivate_ts) > 0.20:
                self.jitter_controller.request_stop()
                logger.info("JITTER STOP")

    # ------------------------------------------------------------------ обработка кадра
    def _process_frame(self, image, t_cap, t_proc):
        """Один кадр: детекция -> замок -> наведение. Возвращает (n_entities, button_held, lock_status)."""
        if self._last_frame_ts:
            self._frame_dt = min(max(t_cap - self._last_frame_ts, 0.001), 0.05)
        self._last_frame_ts = t_cap

        (ref_cx, ref_cy, region_left, region_top, crosshair_x, crosshair_y), current_config = self._geometry()

        debug_img = image.copy() if config.show_debug_window else None
        result = detect(self.model, image, self.class_names, current_config)
        if debug_img is not None:
            draw_debug_overlay(debug_img, result)

        raw_button_held = is_button_pressed(config.selected_mouse_button)

        # Наведение, отправленное с прошлого кадра; на этом кадре его уже учли
        aim_move = (self._last_aim_dx, self._last_aim_dy)
        self._last_aim_dx = self._last_aim_dy = 0.0
        ego_shift = self.ego_tracker.update(aim_move=aim_move, t=t_cap)

        life_result = self.lifecycle.update(
            result.entities, image.shape[:2], raw_button_held,
            ego_shift=ego_shift, t=t_cap,
        )
        best_target = life_result.target
        lock_status = life_result.status
        button_held = life_result.held

        if (getattr(config, "ego_calib_log", False) and lock_status == LockStatus.TRACKING
                and best_target is not None and self._last_best_center is not None):
            logger.info(
                "EGO | dx=%.1f dy=%.1f | mx=%.2f my=%.2f | dt=%.1fms",
                best_target.poi[0] - self._last_best_center[0],
                best_target.poi[1] - self._last_best_center[1],
                aim_move[0], aim_move[1], self._frame_dt * 1000)

        if best_target:
            tx, ty = best_target.poi
            self._last_dist = best_target.distance_px
            self._last_target_ts = t_cap
            if lock_status == LockStatus.TRACKING:
                self._last_best_center = (tx, ty)
                self._had_target = True

            box_height = best_target.height
            if self.ema_target_x is None:
                self.ema_target_x, self.ema_target_y = tx, ty
            else:
                jump_dist = math.hypot(tx - self.ema_target_x, ty - self.ema_target_y)
                # Плавная EMA: чем больше скачок, тем меньше сглаживание
                max_dist = 20.0
                min_alpha, max_alpha = 0.15, 0.85
                raw_alpha = min_alpha + (max_alpha - min_alpha) * min(jump_dist / max_dist, 1.0)
                if box_height < 60 and jump_dist > 2.0:
                    raw_alpha = max(raw_alpha, 0.85)
                alpha = max(min_alpha, min(max_alpha, raw_alpha))
                self.ema_target_x = alpha * tx + (1 - alpha) * self.ema_target_x
                self.ema_target_y = alpha * ty + (1 - alpha) * self.ema_target_y

            target_screen_x = region_left + self.ema_target_x
            target_screen_y = region_top + self.ema_target_y
            mult = self._sens_multiplier()
            raw_dx = (target_screen_x - crosshair_x) * mult
            raw_dy = (target_screen_y - crosshair_y) * mult
            self._apply_movement_mode(raw_dx, raw_dy, 0, best_target)
        else:
            self._had_target = False
            self._last_best_center = None
            if t_cap - self._last_target_ts > 0.20:
                self._reset_idle_state()

        self._process_jitter(button_held)

        # Отладочное окно OpenCV
        if config.show_debug_window and debug_img is not None and result.entities:
            if not self._debug_window_initialized:
                cv2.namedWindow(self._debug_win_name, cv2.WINDOW_AUTOSIZE)
                x = (config.screen_width - debug_img.shape[1]) // 2
                y = (config.screen_height - debug_img.shape[0]) // 2
                cv2.moveWindow(self._debug_win_name, x, y)
                self._debug_window_initialized = True
            cv2.imshow(self._debug_win_name, debug_img)
            cv2.waitKey(1)

        return len(result.entities), button_held, lock_status

    def detection_and_aim_loop(self):
        global fps
        self.model, self.class_names = load_model(config.model_path)
        if self.model is not None:
            logger.info("Warming up model (10 frames)...")
            dummy_image = np.zeros((config.imgsz, config.imgsz, 3), dtype=np.uint8)
            for _ in range(10):
                perform_detection(self.model, dummy_image)
            logger.info("Warmup complete")

        frame_count = 0
        start_time = time.perf_counter()
        n_ent, button_held, lock_status = 0, False, LockStatus.IDLE

        while not self.stop_event.is_set():
            try:
                image, t_cap = self.frame_queue.get(timeout=1)
            except queue.Empty:
                logger.warning("Frame queue is empty. Capture thread may have stalled.")
                continue

            if self.model is None:
                time.sleep(0.1)
                frame_count += 1
                continue

            t_proc = time.perf_counter()
            try:
                n_ent, button_held, lock_status = self._process_frame(image, t_cap, t_proc)
            except Exception as e:  # noqa: BLE001
                if t_proc - self._err_ts > 1.0:
                    logger.exception("Frame processing failed: %s", e)
                    self._err_ts = t_proc
                continue

            # Статистика
            now = time.perf_counter()
            frame_ms = (now - t_proc) * 1000.0
            age_ms = (t_proc - t_cap) * 1000.0
            self._frame_ms_ema = frame_ms if not self._frame_ms_ema else 0.9 * self._frame_ms_ema + 0.1 * frame_ms
            self._age_ms_ema = age_ms if not self._age_ms_ema else 0.9 * self._age_ms_ema + 0.1 * age_ms
            if frame_ms > 35.0:
                self._slow_frames += 1
                logger.debug("SLOW FRAME %.0f ms", frame_ms)

            frame_count += 1
            elapsed = now - start_time
            if elapsed > 1.0:
                self.fps = frame_count / elapsed
                fps = self.fps
                start_time = now
                frame_count = 0

                if now - self._last_summary_ts >= 5.0:
                    st = get_stats()
                    p95 = st.get("p95_ms")
                    logger.info(
                        "STATUS | fps=%.1f frame=%.1fms age=%.1fms infer_p95=%s targets=%d held=%s "
                        "lock=%s lock_waits=%d slow=%d dup_skipped=%d dz_active=%s",
                        self.fps, self._frame_ms_ema, self._age_ms_ema,
                        f"{p95:.1f}ms" if p95 else "n/a",
                        n_ent, button_held, lock_status.value,
                        self._lock_waits, self._slow_frames, self._dup_skipped, self.deadzone.active,
                    )
                    self._last_summary_ts = now
                    self._lock_waits = 0
                    self._slow_frames = 0
                    self._dup_skipped = 0

    # ------------------------------------------------------------------ жизненный цикл
    def start(self):
        global _aimbot_running, makcu
        if self._running:
            return

        if self.makcu is None:
            if makcu is not None:
                self.makcu = makcu
                self._owns_makcu = False
            else:
                try:
                    self.makcu = Mouse()
                    self._owns_makcu = True
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to initialize Mouse: {e}")
                    return
        makcu = self.makcu

        self.stop_event = threading.Event()
        self._reset_runtime_state()
        self._ensure_jitter()          # ДО потоков: ego должен знать про джиттер с первого кадра

        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        self.smooth_thread = threading.Thread(target=self.smooth_movement_loop, daemon=True)
        self.smooth_thread.start()
        self.aim_thread = threading.Thread(target=self.detection_and_aim_loop, daemon=True)
        self.aim_thread.start()

        self._running = True
        _aimbot_running = True

        log_startup_config(
            self.target_lock.config,
            self.detector_config,
            ego=self.ego_tracker,
            logger=logger,
        )
        logger.info(f"Aimbot v{__version__} started in {config.mode} mode.")

    def stop(self):
        global _aimbot_running, makcu
        if not self._running:
            return

        logger.info("Stopping aimbot...")
        self._running = False
        _aimbot_running = False
        self.stop_event.set()

        # Разбудить smooth-поток, блокирующийся на get()
        try:
            self.smooth_move_queue.put_nowait((0.0, 0.0, 0.0))
        except queue.Full:
            pass

        for t in (self.capture_thread, self.smooth_thread, self.aim_thread):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)

        if self.jitter_controller:
            self.jitter_controller.stop()

        if self._owns_makcu and self.makcu is not None:
            try:
                Mouse.cleanup()
            except Exception:  # noqa: BLE001
                pass
            self.makcu = None
            self._owns_makcu = False
            makcu = None

        if config.show_debug_window:
            try:
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass
        logger.info("Aimbot stopped")

    def is_running(self):
        return self._running

    def reload_model(self, path=None):
        self.model, self.class_names = load_model(path or config.model_path)
        return self.model

    def get_model_classes(self, path=None):
        _, class_names = load_model(path or config.model_path)
        return [class_names[i] for i in sorted(class_names.keys())]

    def get_model_size(self, path=None):
        try:
            return f"{os.path.getsize(path or config.model_path) / (1024 * 1024):.2f} MB"
        except Exception:  # noqa: BLE001
            return "?"


# === Глобальные функции управления (API для GUI) ===
_aimbot_instance = None


def start_aimbot():
    global _aimbot_instance
    if _aimbot_instance is None:
        _aimbot_instance = Aimbot()
    _aimbot_instance.start()


def stop_aimbot():
    global _aimbot_instance
    if _aimbot_instance is not None:
        _aimbot_instance.stop()


def is_aimbot_running():
    global _aimbot_instance
    return _aimbot_instance is not None and _aimbot_instance.is_running()


def reload_model(path=None):
    global _aimbot_instance
    if _aimbot_instance is not None:
        return _aimbot_instance.reload_model(path)
    return load_model(path or config.model_path)


def get_model_classes(path=None):
    global _aimbot_instance
    if _aimbot_instance is not None:
        return _aimbot_instance.get_model_classes(path)
    _, class_names = load_model(path or config.model_path)
    return [class_names[i] for i in sorted(class_names.keys())]


def get_model_size(path=None):
    global _aimbot_instance
    if _aimbot_instance is not None:
        return _aimbot_instance.get_model_size(path)
    try:
        return f"{os.path.getsize(path or config.model_path) / (1024 * 1024):.2f} MB"
    except Exception:  # noqa: BLE001
        return "?"


def get_fps():
    global _aimbot_instance
    return _aimbot_instance.fps if _aimbot_instance else 0


__all__ = [
    "start_aimbot", "stop_aimbot", "is_aimbot_running", "reload_model",
    "get_model_classes", "get_model_size", "get_fps",
    "fps", "_aimbot_running", "makcu",
]
