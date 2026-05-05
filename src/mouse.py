import threading
import serial
from serial.tools import list_ports
import time

makcu = None
makcu_lock = threading.Lock()
button_states = {i: False for i in range(5)}
button_states_lock = threading.Lock()
is_connected = False
last_value = 0

SUPPORTED_DEVICES = [
    ("1A86:55D3", "MAKCU"),
    ("1A86:5523", "CH343"),
    ("1A86:7523", "CH340"),
    ("1A86:5740", "CH347"),
    ("10C4:EA60", "CP2102"),
]
BAUD_RATES = [4_000_000, 2_000_000, 115_200]
BAUD_CHANGE_COMMAND = bytearray([0xDE, 0xAD, 0x05, 0x00, 0xA5, 0x00, 0x09, 0x3D, 0x00])

# Глобальное состояние блокировки движений
movement_lock_state = {
    "lock": threading.Lock(),
    "lock_x": False,           # Текущая блокировка оси X
    "lock_y": False,           # Текущая блокировка оси Y
    "aimbot_locked": False,    # Флаг: аимбот активен
    "last_move_time": 0,       # Время последнего движения
    "timeout": 0.1             # Таймаут авто-разблокировки (100 мс)
}

def tick_movement_lock_manager(timeout_override: float = None):
    """
    Менеджер блокировки осей — вызывается в каждом тике основного цикла.
    
    :param timeout_override: Переопределение таймаута (для тестов)
    """
    try:
        # Неблокирующий захват мьютекса (10 мс)
        lock_acquired = movement_lock_state["lock"].acquire(timeout=0.01)
        if not lock_acquired:
            return  # Не удалось захватить — пропустим этот тик
            
        try:
            current_time = time.time()
            timeout = timeout_override if timeout_override is not None else movement_lock_state["timeout"]
            
            # Авто-разблокировка по таймауту бездействия
            if movement_lock_state["aimbot_locked"]:
                idle_time = current_time - movement_lock_state["last_move_time"]
                if idle_time > timeout:
                    # Сбрасываем состояние блокировки
                    movement_lock_state["aimbot_locked"] = False
                    movement_lock_state["lock_x"] = False
                    movement_lock_state["lock_y"] = False
                    # Опционально: лог для отладки
                    # print(f"[MouseLock] Auto-unlocked after {idle_time*1000:.0f}ms idle")
                    
        finally:
            movement_lock_state["lock"].release()
            
    except Exception as e:
        # Не даём ошибке прервать основной цикл
        print(f"[MouseLock] Error in tick: {e}")

# ====================================================================
# Mouse Lock API — публичные функции для управления блокировками
# ====================================================================

def enable_aimbot_lock(lock_x: bool = False, lock_y: bool = False):
    """
    Активировать блокировку для аимбота.
    
    :param lock_x: Блокировать ось X
    :param lock_y: Блокировать ось Y
    """
    try:
        if movement_lock_state["lock"].acquire(timeout=0.01):
            try:
                movement_lock_state["aimbot_locked"] = True
                movement_lock_state["lock_x"] = lock_x
                movement_lock_state["lock_y"] = lock_y
                movement_lock_state["last_move_time"] = time.time()
            finally:
                movement_lock_state["lock"].release()
    except Exception as e:
        print(f"[MouseLock] enable_aimbot_lock error: {e}")

def disable_aimbot_lock():
    """Мгновенно отключить блокировку аимбота"""
    try:
        if movement_lock_state["lock"].acquire(timeout=0.01):
            try:
                movement_lock_state["aimbot_locked"] = False
                movement_lock_state["lock_x"] = False
                movement_lock_state["lock_y"] = False
            finally:
                movement_lock_state["lock"].release()
    except Exception as e:
        print(f"[MouseLock] disable_aimbot_lock error: {e}")

def set_lock_timeout(seconds: float):
    """Установить таймаут авто-разблокировки (в секундах)"""
    with movement_lock_state["lock"]:
        movement_lock_state["timeout"] = max(0.01, min(2.0, seconds))  # clamp 10ms–2s

def get_lock_state() -> dict:
    """Получить текущее состояние блокировок (для GUI/отладки)"""
    with movement_lock_state["lock"]:
        return {
            "aimbot_locked": movement_lock_state["aimbot_locked"],
            "lock_x": movement_lock_state["lock_x"],
            "lock_y": movement_lock_state["lock_y"],
            "last_move_time": movement_lock_state["last_move_time"],
            "timeout": movement_lock_state["timeout"],
        }

def find_com_ports():
    found = []
    for port in list_ports.comports():
        hwid = port.hwid.upper()
        desc = port.description.upper()
        for vidpid, name in SUPPORTED_DEVICES:
            if vidpid in hwid or name.upper() in desc:
                found.append((port.device, name))
                break
    return found

def km_version_ok(ser):
    try:
        ser.reset_input_buffer()
        ser.write(b"km.version()\r")
        ser.flush()
        time.sleep(0.1)
        resp = b""
        start = time.time()
        while time.time() - start < 0.3:
            if ser.in_waiting:
                resp += ser.read(ser.in_waiting)
                if b"km.MAKCU" in resp or b"MAKCU" in resp:
                    return True
            time.sleep(0.01)
        return False
    except Exception as e:
        print(f"[WARN] km_version_ok: {e}")
        return False

def connect_to_makcu():
    global makcu, is_connected
    ports = find_com_ports()
    if not ports:
        print("[ERROR] No supported serial devices found.")
        return False

    for port_name, dev_name in ports:
        if dev_name == "MAKCU":
            for baud in BAUD_RATES:
                print(f"[INFO] Probing MAKCU {port_name} @ {baud} with km.version()...")
                ser = None
                try:
                    ser = serial.Serial(port_name, baud, timeout=0.3)
                    time.sleep(0.1)
                    if km_version_ok(ser):
                        if baud == 115_200:
                            print("[INFO] MAKCU responded at 115200, sending 4M handshake...")
                            ser.write(BAUD_CHANGE_COMMAND)
                            ser.flush()
                            ser.close()
                            time.sleep(0.15)
                            # --- Always cleanup before opening new connection! ---
                            ser4m = None
                            try:
                                ser4m = serial.Serial(port_name, 4_000_000, timeout=0.3)
                                time.sleep(0.1)
                                if km_version_ok(ser4m):
                                    print(f"[INFO] MAKCU handshake successful, switching to 4M on {port_name}.")
                                    ser4m.close()
                                    time.sleep(0.1)
                                    makcu = serial.Serial(port_name, 4_000_000, timeout=0.1)
                                    with makcu_lock:
                                        makcu.write(b"km.buttons(1)\r")
                                        makcu.flush()
                                    is_connected = True
                                    return True
                                else:
                                    print("[WARN] 4M handshake failed, staying at 115200.")
                                    ser4m.close()
                                    time.sleep(0.1)
                                    makcu = serial.Serial(port_name, 115_200, timeout=0.1)
                                    with makcu_lock:
                                        makcu.write(b"km.buttons(1)\r")
                                        makcu.flush()
                                    is_connected = True
                                    return True
                            except Exception as e:
                                print(f"[WARN] Could not switch to 4M: {e}")
                                if ser4m:
                                    try:
                                        ser4m.close()
                                    except:
                                        pass
                                time.sleep(0.1)
                                makcu = serial.Serial(port_name, 115_200, timeout=0.1)
                                with makcu_lock:
                                    makcu.write(b"km.buttons(1)\r")
                                    makcu.flush()
                                is_connected = True
                                return True
                        else:
                            print(f"[INFO] MAKCU responded at {baud}, using it.")
                            ser.close()
                            time.sleep(0.1)
                            makcu = serial.Serial(port_name, baud, timeout=0.1)
                            with makcu_lock:
                                makcu.write(b"km.buttons(1)\r")
                                makcu.flush()
                            is_connected = True
                            return True
                    ser.close()
                    time.sleep(0.1)
                except Exception as e:
                    print(f"[WARN] Failed MAKCU@{baud}: {e}")
                    if ser:
                        try:
                            ser.close()
                        except:
                            pass
                        time.sleep(0.1)
                    if makcu and makcu.is_open:
                        makcu.close()
                    makcu = None
                    is_connected = False
        else:
            for baud in BAUD_RATES:
                print(f"[INFO] Trying {dev_name} {port_name} @ {baud} ...")
                ser = None
                try:
                    ser = serial.Serial(port_name, baud, timeout=0.1)
                    with makcu_lock:
                        ser.write(b"km.buttons(1)\r")
                        ser.flush()
                    ser.close()
                    time.sleep(0.1)
                    makcu = serial.Serial(port_name, baud, timeout=0.1)
                    is_connected = True
                    print(f"[INFO] Connected to {dev_name} on {port_name} at {baud} baud.")
                    return True
                except Exception as e:
                    print(f"[WARN] Failed {dev_name}@{baud}: {e}")
                    if ser:
                        try:
                            ser.close()
                        except:
                            pass
                        time.sleep(0.1)
                    if makcu and makcu.is_open:
                        makcu.close()
                    makcu = None
                    is_connected = False

    print("[ERROR] Could not connect to any supported device.")
    return False


def count_bits(n: int) -> int:
    return bin(n).count("1")

def listen_makcu():
    global last_value
    # start from a clean state
    last_value = 0
    with button_states_lock:
        for i in range(5):
            button_states[i] = False

    while is_connected:
        try:
            b = makcu.read(1)  # blocking read (uses port timeout)
            if not b:
                continue

            v = b[0]

            # Ignore echoed ASCII (including CR/LF). Only 0..31 are valid masks.
            if v in (0x0A, 0x0D) or v > 31:
                continue

            # v is a 5-bit mask (bit0..bit4). Update only changed bits.
            changed = last_value ^ v
            if changed:
                with button_states_lock:
                    for i in range(5):
                        m = 1 << i
                        if changed & m:
                            button_states[i] = bool(v & m)
                last_value = v

        except serial.SerialException as e:
            print(f"[ERROR] Listener serial exception: {e}")
            break
        except Exception as e:
            # swallow transient errors but keep running
            print(f"[WARN] Listener error: {e}")
            time.sleep(0.001)

    # ensure clean state on exit
    with button_states_lock:
        for i in range(5):
            button_states[i] = False
    last_value = 0

def is_button_pressed(idx: int) -> bool:
    with button_states_lock:
        return button_states.get(idx, False)

def test_move():
    if is_connected:
        with makcu_lock:
            makcu.write(b"km.move(100,100)\r")
            makcu.flush()

# --------------------------------------------------------------------
# Button Lock / Masking helpers
# --------------------------------------------------------------------

# Index mapping: 0=L, 1=R, 2=M, 3=S4, 4=S5
_LOCK_CMD_BY_IDX = {
    0: "lock_ml",
    1: "lock_mr",
    2: "lock_mm",
    3: "lock_ms1",
    4: "lock_ms2",
}

# State tracked by mask manager (so we only send lock/unlock when needed)
_mask_applied_idx = None

def _send_cmd_no_wait(cmd: str):
    """Send 'km.<cmd>\\r' without waiting for response (listener ignores ASCII)."""
    if not is_connected:
        return
    with makcu_lock:
        makcu.write(f"km.{cmd}\r".encode("ascii", "ignore"))
        makcu.flush()

def lock_button_idx(idx: int):
    """Lock a single button by index (0..4)."""
    cmd = _LOCK_CMD_BY_IDX.get(idx)
    if cmd is None:
        return
    _send_cmd_no_wait(f"{cmd}(1)")

def unlock_button_idx(idx: int):
    """Unlock a single button by index (0..4)."""
    cmd = _LOCK_CMD_BY_IDX.get(idx)
    if cmd is None:
        return
    _send_cmd_no_wait(f"{cmd}(0)")

def unlock_all_locks():
    """Best-effort unlock of all lockable buttons."""
    for i in range(5):
        unlock_button_idx(i)

def mask_manager_tick(selected_idx: int, aimbot_running: bool):
    """Manage button locks based on selected_idx and aimbot_running state."""
    
    global _mask_applied_idx

    if not is_connected:
        _mask_applied_idx = None
        return

    # clamp invalid index
    if not isinstance(selected_idx, int) or not (0 <= selected_idx <= 4):
        selected_idx = None

    if not aimbot_running:
        if _mask_applied_idx is not None:
            unlock_button_idx(_mask_applied_idx)
            _mask_applied_idx = None
        return

    # running: apply lock for selected_idx
    if selected_idx is None:
        # nothing selected -> make sure nothing is locked
        if _mask_applied_idx is not None:
            unlock_button_idx(_mask_applied_idx)
            _mask_applied_idx = None
        return

    if _mask_applied_idx != selected_idx:
        # switch lock to a new button
        if _mask_applied_idx is not None:
            unlock_button_idx(_mask_applied_idx)
        lock_button_idx(selected_idx)
        _mask_applied_idx = selected_idx

class Mouse:
    _instance = None
    _listener = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_inited"):
            if not connect_to_makcu():
                print("[ERROR] Mouse init failed to connect.")
            else:
                Mouse._listener = threading.Thread(target=listen_makcu, daemon=True)
                Mouse._listener.start()
            self._inited = True

    def move(self, x: float, y: float):
        """Отправка движения с учётом блокировок осей (Mouse Lock)"""
        if not is_connected:
            return
            
        # Применяем блокировки осей из movement_lock_state
        if movement_lock_state["aimbot_locked"]:
            if movement_lock_state["lock_x"]:
                x = 0
            if movement_lock_state["lock_y"]:
                y = 0
                
        # Если оба значения нулевые — нет смысла отправлять команду
        if x == 0 and y == 0:
            # Но всё равно обновим last_move_time, чтобы таймаут не сработал ложно
            with movement_lock_state["lock"]:
                movement_lock_state["last_move_time"] = time.time()
            return
            
        dx, dy = int(x), int(y)
        
        with makcu_lock:
            makcu.write(f"km.move({dx},{dy})\r".encode())
            makcu.flush()
            
        # Обновляем время последнего движения для авто-разблокировки
        with movement_lock_state["lock"]:
            movement_lock_state["last_move_time"] = time.time()
    def move_bezier(self, x: float, y: float, segments: int, ctrl_x: float, ctrl_y: float):
        if not is_connected:
            return
            
        # 🔒 Применяем блокировки осей (аналогично move())
        if movement_lock_state["aimbot_locked"]:
            if movement_lock_state["lock_x"]:
                x = 0
            if movement_lock_state["lock_y"]:
                y = 0
                
        with makcu_lock:
            cmd = f"km.move({int(x)},{int(y)},{int(segments)},{int(ctrl_x)},{int(ctrl_y)})\r"
            makcu.write(cmd.encode())
            makcu.flush()

    def click(self):
        if not is_connected:
            return
        with makcu_lock:
            makcu.write(b"km.left(1)\r")
            makcu.flush()
            makcu.write(b"km.left(0)\r")
            makcu.flush()

    @staticmethod
    def mask_manager_tick(selected_idx: int, aimbot_running: bool):
        """Static wrapper so callers can do: Mouse.mask_manager_tick(idx, running)."""
        mask_manager_tick(selected_idx, aimbot_running)

    @staticmethod
    def cleanup():
        global is_connected, makcu, _mask_applied_idx
        
        # Сброс состояния Mouse Lock
        try:
            with movement_lock_state["lock"]:
                movement_lock_state["aimbot_locked"] = False
                movement_lock_state["lock_x"] = False
                movement_lock_state["lock_y"] = False
        except Exception:
            pass

        # Всегда снимаем блокировки кнопок перед закрытием порта
        try:
            unlock_all_locks()
        except Exception:
            pass
        _mask_applied_idx = None

        is_connected = False
        if makcu and makcu.is_open:
            makcu.close()
        Mouse._instance = None
        Mouse._listener = None
        print("[INFO] Mouse serial cleaned up.")
