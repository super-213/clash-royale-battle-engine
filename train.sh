#!/bin/zsh
set -eu

SCRIPT_DIR=${0:A:h}
CONDA_ENV_NAME=${CLASH_TRAIN_CONDA_ENV:-clash-royale}
ANDROID_SDK_DIR=${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/Library/Android/sdk}}
ADB_BIN=$ANDROID_SDK_DIR/platform-tools/adb
EMULATOR_BIN=$ANDROID_SDK_DIR/emulator/emulator
AVD_NAME=${CLASH_TRAIN_AVD:-Pixel_10_Pro}
EMULATOR_SERIAL=${CLASH_TRAIN_SERIAL:-emulator-5554}

cd "$SCRIPT_DIR"
if ! conda run -n "$CONDA_ENV_NAME" python -c "import sb3_contrib, frida" >/dev/null 2>&1; then
  print -u2 "缺少训练依赖 sb3-contrib 或 frida。请运行："
  print -u2 "conda run -n $CONDA_ENV_NAME python -m pip install -r requirements-training.txt"
  exit 1
fi

# Configuration checks and help do not need an Android runtime.
for argument in "$@"; do
  if [[ "$argument" == --dry-run || "$argument" == --help || "$argument" == -h ]]; then
    exec conda run --no-capture-output -n "$CONDA_ENV_NAME" python -u one_click_train.py "$@"
  fi
done

if [[ ! -x "$ADB_BIN" || ! -x "$EMULATOR_BIN" ]]; then
  print -u2 "找不到 Android SDK 的 adb 或 emulator：$ANDROID_SDK_DIR"
  print -u2 "请设置 ANDROID_SDK_ROOT 后重试。"
  exit 1
fi
export PATH="${ADB_BIN:h}:$PATH"
export CLASH_TRAIN_SERIAL="$EMULATOR_SERIAL"

EMULATOR_PID=""
DEVICE_STATE=$("$ADB_BIN" devices | awk -v serial="$EMULATOR_SERIAL" '$1 == serial {print $2}')
if [[ -z "$DEVICE_STATE" ]]; then
  if ! "$EMULATOR_BIN" -list-avds | grep -Fxq "$AVD_NAME"; then
    print -u2 "找不到 AVD：$AVD_NAME（可用 CLASH_TRAIN_AVD 指定已有 AVD）"
    exit 1
  fi
  mkdir -p artifacts/emulator
  EMULATOR_LOG="$SCRIPT_DIR/artifacts/emulator/$AVD_NAME.log"
  print "[emulator] 启动 $AVD_NAME，日志：$EMULATOR_LOG"
  EMULATOR_PID=$(python3 - "$EMULATOR_BIN" "$AVD_NAME" "$EMULATOR_LOG" <<'PY'
import subprocess
import sys

with open(sys.argv[3], "a", encoding="utf-8") as log:
    process = subprocess.Popen(
        [sys.argv[1], "-avd", sys.argv[2], "-no-window", "-gpu", "swiftshader_indirect", "-no-audio"],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print(process.pid)
PY
  )
fi
for (( attempt = 0; attempt < 120; attempt++ )); do
  if [[ "$("$ADB_BIN" -s "$EMULATOR_SERIAL" get-state 2>/dev/null || true)" == device ]] && \
     [[ "$("$ADB_BIN" -s "$EMULATOR_SERIAL" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" == 1 ]]; then
    break
  fi
  if [[ -n "$EMULATOR_PID" ]] && ! kill -0 "$EMULATOR_PID" 2>/dev/null; then
    print -u2 "模拟器启动失败，请查看 $EMULATOR_LOG"
    exit 1
  fi
  sleep 1
done
if (( attempt == 120 )); then
  print -u2 "等待模拟器 $EMULATOR_SERIAL 启动超时"
  exit 1
fi
print "[emulator] $EMULATOR_SERIAL 已就绪"
exec conda run --no-capture-output -n "$CONDA_ENV_NAME" python -u one_click_train.py "$@"
