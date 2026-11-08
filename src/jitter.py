import bisect
import math
import sys
import threading
import time
import logging
from collections import deque, namedtuple
from config import config

log = logging.getLogger("jitter")

PATTERNS = {
    # шаги в единицах амплитуды: (dx, dy)
    'square':           [(-1, 0), (0, 1), (1, 0), (0, -1)],
    'downward_zigzag':  [(-1, 1), (1, 1)],
    'diamond':          [(-1, -1), (1, -1), (1, 1), (-1, 1)],
    'horizontal_shake': [(-1, 0), (2, 0), (-1, 0)],
}
DEFAULT_PATTERN = 'square'

MIN_SPEED_MS = 1          # speed_ms <= 0 превратил бы цикл в hot loop без пауз
STOP_JOIN_TIMEOUT = 0.5   # сколько stop()/start() ждут завершения потока
ROLLBACK_TIMEOUT = 0.3    # сколько откат ждёт lock устройства
LOCK_POLL = 0.05          # шаг опроса lock, чтобы вовремя замечать stop
START_JOIN_TIMEOUT = 0.05  # сколько start() ждёт ещё не вышедший поток (вызывается из цикла кадров:
#                            ждать долго нельзя; не успел - вернёт False, повторите на следующем кадре)
OFFSET_HISTORY = 1024     # записей (время, смещение) для offset_at(): ~3 с при шаге 3 мс

# Неизменяемый снимок параметров: update_params подменяет его одним присваиванием
Params = namedtuple('Params', 'amp_x amp_y speed_ms pattern')


def _check_amp(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"jitter: {name} must be a finite number, got {value!r}")
    return value


def _check_speed(value):
    value = _check_amp('speed_ms', value)
    if value < MIN_SPEED_MS:
        log.warning("jitter: speed_ms=%r is too small, using %d", value, MIN_SPEED_MS)
        return MIN_SPEED_MS
    return value


def _round_half_away(v):
    """Округление 0.5 от нуля (симметрично для +/-, в отличие от банковского round)."""
    return int(math.copysign(math.floor(abs(v) + 0.5), v))


def _from_config(key, default, check):
    raw = getattr(config, key, default)
    try:
        return check(raw)
    except ValueError:
        log.warning("jitter: bad config %s=%r, using %r", key, raw, default)
        return default


# Python 3.11+ на Windows сам спит по высокоточному таймеру (CREATE_WAITABLE_TIMER_HIGH_RESOLUTION):
# time.sleep(0.003) держит ~3 мс без timeBeginPeriod. Этот таймер не зависит от того, свёрнуто ли
# окно (Windows 11 игнорирует timeBeginPeriod у невидимых процессов) и не меняет системный тик.
# На 3.10 и старше остаётся timeBeginPeriod(1). Переопределить: JITTER_FORCE_TIMER_PERIOD = True.
HIGH_RES_SLEEP = sys.version_info >= (3, 11)
JITTER_FORCE_TIMER_PERIOD = False

if sys.platform == 'win32':
    import ctypes
    _winmm = ctypes.WinDLL('winmm')

    def _timer_begin():
        # Без этого шаг sleep на Windows (Python < 3.11) ~15 мс, а не 1 мс.
        # Флаг процессный, а не потоковый: влияет на энергопотребление всего приложения.
        try:
            return _winmm.timeBeginPeriod(1) == 0  # 0 == TIMERR_NOERROR
        except Exception:
            return False

    def _timer_end():
        _winmm.timeEndPeriod(1)
else:
    def _timer_begin():
        return True

    def _timer_end():
        pass


class JitterController:
    def __init__(self, makcu_instance, makcu_lock=None):
        self.makcu = makcu_instance
        self.lock = makcu_lock or threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._params_write_lock = threading.Lock()
        self._warned_pattern = None
        # Накопленное смещение, которое джиттер РЕАЛЬНО отправил в устройство (единицы мыши).
        # Пишет только поток джиттера (шаги и откат), читают любые потоки: присваивание кортежа атомарно.
        self._offset = (0.0, 0.0)
        self._hist = deque([(time.perf_counter(), 0.0, 0.0)], maxlen=OFFSET_HISTORY)
        # Дробная часть шага, не отправленная в устройство (переносится на следующий шаг):
        # в устройство уходят только целые единицы, поэтому откат и offset точные
        self._carry = (0.0, 0.0)
        self._cfg_seen = None          # значения config при последнем чтении (см. start())
        self.update_params()

    @property
    def is_active(self):
        return self._thread is not None and self._thread.is_alive()

    @is_active.setter
    def is_active(self, value):
        # legacy: jitter.is_active = True/False. False теперь НЕ блокирует (request_stop);
        # True вызывает start() (ждёт не дольше START_JOIN_TIMEOUT)
        self.start() if value else self.request_stop()

    @property
    def offset(self):
        """(x, y): сколько единиц мыши джиттер сдвинул в сумме с момента создания объекта
        (шаги и откаты). Это КОМАНДА, отправленная в устройство; если оно округляет дробные
        шаги, фактическое смещение может отличаться."""
        return self._offset

    def offset_at(self, t):
        """Смещение (x, y) на момент t (time.perf_counter()), например на момент съёмки кадра
        минус задержка захвата. Для t раньше самой старой записи - самая старая запись."""
        hist = self._hist.copy()      # снимок одной C-операцией: поток джиттера может дописывать
        i = bisect.bisect_right(hist, t, key=lambda r: r[0])   # после последней записи с меткой <= t
        _, x, y = hist[max(i - 1, 0)]
        return x, y

    def _record(self, dx, dy, t):
        ox, oy = self._offset
        ox, oy = ox + dx, oy + dy
        self._offset = (ox, oy)
        self._hist.append((t, ox, oy))

    # Параметры раньше были обычными атрибутами, которые меняли на лету;
    # цикл подхватывает изменение со следующего цикла паттерна.
    # Сеттеры можно вызывать из любых потоков (запись под _params_write_lock).
    def _set_param(self, **kw):
        with self._params_write_lock:
            self._params = self._params._replace(**kw)

    @property
    def amp_x(self):
        return self._params.amp_x

    @amp_x.setter
    def amp_x(self, value):
        self._set_param(amp_x=_check_amp('amp_x', value))

    @property
    def amp_y(self):
        return self._params.amp_y

    @amp_y.setter
    def amp_y(self, value):
        self._set_param(amp_y=_check_amp('amp_y', value))

    @property
    def speed_ms(self):
        return self._params.speed_ms

    @speed_ms.setter
    def speed_ms(self, value):
        self._set_param(speed_ms=_check_speed(value))

    @property
    def pattern(self):
        return self._params.pattern

    @pattern.setter
    def pattern(self, value):
        self._set_param(pattern=value)

    @staticmethod
    def _read_config():
        return Params(
            _from_config('jitter_amp_x', 3, lambda v: _check_amp('amp_x', v)),
            _from_config('jitter_amp_y', 3, lambda v: _check_amp('amp_y', v)),
            _from_config('jitter_speed_ms', 3, _check_speed),
            getattr(config, 'jitter_pattern', 'square'),
        )

    def update_params(self):
        """Полностью перечитать параметры из config (значения из сеттеров заменяются)."""
        params = self._read_config()
        with self._params_write_lock:
            self._params = params
            self._cfg_seen = params

    def _refresh_from_config(self):
        """Взять из config только то, что ИЗМЕНИЛОСЬ в нём с прошлого чтения. Значения,
        заданные сеттерами (j.amp_x = ...), сохраняются, пока соответствующий ключ config
        не поменяют."""
        cfg = self._read_config()
        with self._params_write_lock:
            seen = self._cfg_seen
            if seen is None:
                self._params = cfg
            else:
                changed = {f: getattr(cfg, f) for f in Params._fields
                           if getattr(cfg, f) != getattr(seen, f)}
                if changed:
                    self._params = self._params._replace(**changed)
            self._cfg_seen = cfg

    def _resolve_steps(self, pattern):
        steps = PATTERNS.get(pattern)
        if steps is None:
            if self._warned_pattern != pattern:
                log.warning("Unknown jitter pattern %r, using %r", pattern, DEFAULT_PATTERN)
                self._warned_pattern = pattern
            steps = PATTERNS[DEFAULT_PATTERN]
        return steps

    def _move(self, dx, dy, stop=None, timeout=None):
        """True, если движение выполнено. False, если не дождались lock:
        из-за stop или по timeout. Тогда смещение НЕ произошло."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.lock.acquire(timeout=LOCK_POLL):
            if stop is not None and stop.is_set():
                return False
            if deadline is not None and time.monotonic() >= deadline:
                return False
        try:
            t0 = time.perf_counter()
            self.makcu.move(dx, dy)
            # только после успешной отправки; время - середина вызова (точнее для latency_s)
            self._record(dx, dy, 0.5 * (t0 + time.perf_counter()))
        finally:
            self.lock.release()
        return True

    def _rollback(self, ox, oy):
        if not (ox or oy):
            return                    # цикл завершён целиком: остаток переносится в следующий
        self._carry = (0.0, 0.0)      # прерванный цикл: откатываем и начинаем с чистого остатка
        try:
            if not self._move(-ox, -oy, timeout=ROLLBACK_TIMEOUT):
                log.warning("jitter: rollback skipped, device lock is busy (offset %r, %r)", ox, oy)
        except Exception as e:
            log.warning("jitter: rollback failed: %s", e)

    def _loop(self, stop):
        timer_ok = False
        try:
            if JITTER_FORCE_TIMER_PERIOD or not HIGH_RES_SLEEP:
                timer_ok = _timer_begin()
                if not timer_ok:
                    log.warning("jitter: timeBeginPeriod failed, timing may be coarse (~15 ms)")
            next_t = time.perf_counter()
            while not stop.is_set():
                if not self.makcu:
                    stop.wait(0.1)
                    next_t = time.perf_counter()
                    continue
                p = self._params
                delay = p.speed_ms / 1000.0
                ox = oy = 0  # смещение внутри текущего цикла паттерна
                failed = False
                try:
                    for sx, sy in self._resolve_steps(p.pattern):
                        if stop.is_set():
                            break
                        # дробные амплитуды: шлём целое, остаток переносим на следующий шаг
                        cx, cy = self._carry
                        wx, wy = sx * p.amp_x + cx, sy * p.amp_y + cy
                        dx, dy = _round_half_away(wx), _round_half_away(wy)
                        if dx or dy:
                            if not self._move(dx, dy, stop):
                                break  # не дождались lock из-за stop: движения не было
                            ox += dx
                            oy += dy
                        self._carry = (wx - dx, wy - dy)
                        # дедлайн, а не sleep(delay): время move() не накапливается
                        next_t += delay
                        wait = next_t - time.perf_counter()
                        if wait > 0.05:
                            stop.wait(wait)
                        elif wait > 0:
                            time.sleep(wait)
                        else:
                            next_t = time.perf_counter()  # отстали, не наверстываем
                    else:
                        ox = oy = 0  # цикл завершён целиком, откатывать нечего
                except Exception:
                    failed = True
                    log.exception("WebShaker error")
                finally:
                    # остановка или ошибка посреди цикла: откатываем недоделанную часть
                    self._rollback(ox, oy)
                if failed:
                    stop.wait(0.5)
                    next_t = time.perf_counter()
        finally:
            if timer_ok:
                _timer_end()

    def start(self):
        """True, если поток работает после вызова; False, если запуск отказан
        (предыдущий поток ещё не завершился после stop())."""
        with self._lifecycle_lock:
            t = self._thread
            if t is not None and t.is_alive():
                if not self._stop.is_set():
                    return True  # уже работает
                if t is not threading.current_thread():
                    t.join(timeout=START_JOIN_TIMEOUT)   # коротко: звать из цикла кадров
                if t.is_alive():
                    log.warning("jitter: previous thread still stopping, start refused "
                                "(повторите start() на следующем кадре)")
                    return False
            self._refresh_from_config()             # сеттеры до start() больше не затираются
            self._carry = (0.0, 0.0)
            self._stop = threading.Event()          # новый Event на каждый запуск
            self._thread = threading.Thread(
                target=self._loop, args=(self._stop,), daemon=True, name="JitterThread")
            self._thread.start()
            return True

    def request_stop(self):
        """НЕ блокирует: только просит поток остановиться (Event.set) и сразу возвращается.
        Звать из цикла кадров. Поток сам откатит недоделанный цикл и выйдет (обычно за мс).
        Возвращает True, если поток уже не работает. Не берёт _lifecycle_lock: не зависнет,
        даже если другой поток сейчас ждёт join в stop()."""
        self._stop.set()
        t = self._thread
        return t is None or not t.is_alive()

    def stop(self, wait=True):
        """wait=True: True, если поток завершился; False, если не успел за STOP_JOIN_TIMEOUT
        (или stop() вызван из самого потока: остановка запрошена, но выход не гарантирован).
        wait=False: то же, что request_stop() (не блокирует)."""
        if not wait:
            return self.request_stop()
        with self._lifecycle_lock:
            self._stop.set()
            t = self._thread
            if t is None:
                return True
            if t is threading.current_thread():
                return False
            t.join(timeout=STOP_JOIN_TIMEOUT)
            if t.is_alive():
                log.warning("jitter: thread did not finish in %.1fs", STOP_JOIN_TIMEOUT)
                return False
            return True
